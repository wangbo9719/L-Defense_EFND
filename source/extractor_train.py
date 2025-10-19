import argparse
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from source.dataset import get_raw_datasets, NewsDataset
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import RobertaConfig, RobertaTokenizer

from help import *
from tensorboardX import SummaryWriter
from source.extractor_model import EvidenceSelection

import json


MODEL_CLASSES = {
    'roberta': [RobertaConfig, RobertaTokenizer, EvidenceSelection],
}

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def collect_predictions(logits, margin=0.1):
    logits = logits.detach().cpu().numpy()
    res = logits[:, 1] - logits[:, 0]
    prediction = np.ones(res.shape)  # 1: half-true
    prediction = np.where(res > margin, 2, prediction)   # 2: true
    prediction = np.where(res < -margin, 0, prediction)  # 0: false
    return prediction

def train(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=None):
    """ Train the model """
    eval_fn = eval_fn
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    metric_best = -1e-5

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    assert args.train_batch_size % (args.n_gpu * args.gradient_accumulation_steps) == 0
    batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=train_dataset.data_collate_fn)

    # ------------------- learning step ------------------
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.t_total = t_total

    # ---------------------- set optimizer and scheduler ----------------------
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.adam_betas is not None:
        adam_betas = tuple(float(_f) for _f in args.adam_betas.split(","))
        assert len(adam_betas) == 2
    else:
        adam_betas = (0.9, 0.999)

    warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=adam_betas, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader)*args.num_train_epochs)
    # ------------------------------------------------------------------------------

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    for _idx_epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps), disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch_tensor, batch_text = tuple(t.to(args.device) for t in batch[0]), batch[1]


            stage1_outputs = model(batch_tensor[0], batch_tensor[1], batch_tensor[2], batch_tensor[3], batch_tensor[4], batch_tensor[5], batch_tensor[-1])
            loss, kl_loss, cls_loss, cls_logits, predict_scores, true_scores, false_scores, true_attn_weights, false_attn_weights = stage1_outputs


            # -------- loss backward -----------
            loss.backward()
            tr_loss += loss.item()

            # ----------------------------------

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0.001:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)
                tb_writer.add_scalar('kl_loss', kl_loss.item(), global_step)
                tb_writer.add_scalar('cls_loss', cls_loss.item(), global_step)

                global_step += 1

                # logging
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    logging.info(
                        f'Ep[{_idx_epoch}] Loss: {loss:.4f}\t KL-Loss: {kl_loss:.4f}\t CLS_loss: {cls_loss:.4f}')
                    with open(os.path.join(args.output_dir, 'loss_record.txt'), 'a') as wf:
                        wf.write('ep:' + str(_idx_epoch) + ', global_step:' + str(global_step) + ': ')
                        wf.write(f'Loss: {loss:.4f}\t KL-Loss: {kl_loss:.4f}\t CLS_loss: {cls_loss:.4f}')
                        wf.write('\n')

                # save checkpoint
                if args.save_checkpoint and args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args.output_dir + '/checkpoints/' + str(global_step), model, tokenizer, args)

                # evaluation
                if args.local_rank in [-1, 0] and eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0 :
                    results = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'a') as wf:
                        wf.write('ep' + str(_idx_epoch) + ', global_step:'+ str(global_step) + ': \n')
                        for key, value in results.items():
                            wf.write(key + ':' + f'{100*value:.2f}\n')
                        wf.write('\n')
                    if results['f1'] > metric_best:
                        save_model_with_default_name(args.output_dir, model, tokenizer, args)
                        metric_best = results['f1']

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # can add epoch evaluation
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # save epoch checkpoint
        if args.save_checkpoint:
            save_model_with_default_name(args.output_dir + '/checkpoints/' + str(_idx_epoch), model, tokenizer, args)



    # eval at the end of training
    results = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step)
    for key, value in results.items():
        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    with open(os.path.join(args.output_dir, 'eval_results.txt'), 'a') as wf:
        wf.write('End: \n')
        for key, value in results.items():
            wf.write(key + ':' + f'{100 * value:.2f}\n')
        wf.write('\n')
    if results['f1'] > metric_best:
        save_model_with_default_name(args.output_dir, model, tokenizer, args)
        metric_best = results['f1']


    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return metric_best

def evaluation(args, eval_dataset, model, tokenizer, global_step=None, mode='eval', dataset_type = None):

    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.data_collate_fn)

    model.eval()

    eval_loss, eval_extracted_loss, eval_cls_loss = 0, 0, 0
    iter_count = 0

    predictions = []
    sent_pre_list, sent_recall_list, sent_f1_list = [], [], []
    evidence_num_ratio_list = []
    hits_count = 0

    # record temp evidence details
    tmp_evidence_dict = {key: {} for key in eval_dataset.event_id}

    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        iter_count += 1

        batch_tensor, batch_text = tuple(t.to(args.device) for t in batch[0]), batch[1]

        with torch.no_grad():
           stage1_outputs = model(batch_tensor[0], batch_tensor[1], batch_tensor[2], batch_tensor[3], batch_tensor[4],
                                  batch_tensor[5], batch_tensor[-1])
           loss, kl_loss, cls_loss, cls_logits, predict_scores, true_scores, false_scores, true_attn_weights, false_attn_weights = stage1_outputs

        eval_loss += loss.mean().item()

        # veracity
        cls_logits = torch.softmax(cls_logits, dim=-1)
        batch_predictions = torch.argmax(cls_logits, dim=-1)
        predictions.append(batch_predictions.detach().cpu().numpy())

        # sentence evidence labels
        batch_sents_labels = []
        for claim_dict in batch[1]:
            batch_sents_labels.append(np.array(claim_dict['sents_labels']))

        # extracted scores and predictions
        # use scores to select evidence sentences
        for _i in range(len(batch[1])):
            if batch_predictions[_i] == 0:
                sample_predict_sent_scores = false_scores[_i]
            elif batch_predictions[_i] == 1:
                sample_predict_sent_scores = (true_scores[_i] + false_scores[_i])
            elif batch_predictions[_i] == 2:
                sample_predict_sent_scores = true_scores[_i]
            else:
                raise ValueError('batch_predictions error')


            sample_sorted_predict_sent_scores, sample_sorted_predict_sent_idx = torch.sort(sample_predict_sent_scores,
                                                                                           descending=True)
            # select top-k evidence sentences
            selected_num = min(args.top_k, len(sample_sorted_predict_sent_idx))

            sample_sorted_predict_sent_idx = sample_sorted_predict_sent_idx[:selected_num].detach().cpu().numpy()
            sample_predicted = np.zeros(len(sample_predict_sent_scores))
            sample_predicted[sample_sorted_predict_sent_idx] = 1

            sample_pre = precision_score(batch_sents_labels[_i], sample_predicted, average="binary", pos_label=1)
            sample_recall = recall_score(batch_sents_labels[_i], sample_predicted, average="binary", pos_label=1)
            sample_f1 = f1_score(batch_sents_labels[_i], sample_predicted, average="binary", pos_label=1)
            sent_pre_list.append(sample_pre)
            sent_recall_list.append(sample_recall)
            sent_f1_list.append(sample_f1)


            if sum(batch_sents_labels[_i][sample_sorted_predict_sent_idx]):
                hits_count += 1

            #  ---------------- record temp evidence details -----------------
            if args.get_evidences:
                sent_false_scores, sent_false_idx = torch.sort(false_scores[_i], descending=True)
                false_selected_num = min(args.top_k, len(false_scores))

                sent_false_idx = sent_false_idx[:false_selected_num].detach().cpu().numpy()
                sent_false_scores = sent_false_scores[:false_selected_num].detach().cpu().numpy()
                sent_false_scores = [round(score, 3) for score in (10000 * sent_false_scores).tolist()]
                sent_false_labels = batch_sents_labels[_i][sent_false_idx]
                false_evidence_text = []
                for _j in sent_false_idx:
                    false_evidence_text.append(batch_text[_i]['sents'][_j])

                sent_true_scores, sent_true_idx = torch.sort(true_scores[_i], descending=True)
                true_selected_num = min(args.top_k, len(true_scores))

                sent_true_idx = sent_true_idx[:true_selected_num].detach().cpu().numpy()
                sent_true_scores = sent_true_scores[:true_selected_num].detach().cpu().numpy()
                sent_true_scores = [round(score, 3) for score in (10000 * sent_true_scores).tolist()]
                sent_true_labels = batch_sents_labels[_i][sent_true_idx]
                true_evidence_text = []
                for _j in sent_true_idx:
                    true_evidence_text.append(batch_text[_i]['sents'][_j])


                zipped_true_evidence_details = zip(true_evidence_text, sent_true_scores, sent_true_labels.tolist(), sent_true_idx.tolist())
                true_evidence_details = [[a, b, c, d] for a, b, c, d in zipped_true_evidence_details]

                zipped_false_evidence_details = zip(false_evidence_text, sent_false_scores, sent_false_labels.tolist(), sent_false_idx.tolist())
                false_evidence_details = [[a, b, c, d] for a, b, c, d in zipped_false_evidence_details]

                tmp_evidence_dict[batch[1][_i]['event_id']]['claim'] = batch[1][_i]['claim']
                tmp_evidence_dict[batch[1][_i]['event_id']]['label'] = int(batch[0][2][_i])

                tmp_evidence_dict[batch[1][_i]['event_id']]['predicted_label'] = int(batch_predictions[_i].detach().cpu().numpy())
                tmp_evidence_dict[batch[1][_i]['event_id']]['predicted_score'] = ["{:.3f}".format(e) for e in list(predict_scores[_i].detach().cpu().numpy())]
                tmp_evidence_dict[batch[1][_i]['event_id']]['logsoftmax_predicted_score'] = ["{:.3f}".format(e) for e in list(torch.softmax(predict_scores[_i], dim=0).detach().cpu().numpy())]
                tmp_evidence_dict[batch[1][_i]['event_id']]['explain'] = batch[1][_i]['explain']
                tmp_evidence_dict[batch[1][_i]['event_id']]['true_details'] = true_evidence_details
                tmp_evidence_dict[batch[1][_i]['event_id']]['false_details'] = false_evidence_details
                tmp_evidence_dict[batch[1][_i]['event_id']]['true_idx'] = sent_true_idx.tolist()
                tmp_evidence_dict[batch[1][_i]['event_id']]['false_idx'] = sent_false_idx.tolist()
                tmp_evidence_dict[batch[1][_i]['event_id']]['num_overlap'] = [len(sent_true_idx), len(sent_false_idx), len(set(sent_true_idx) | set(sent_false_idx))]

            # ---------------------------------------------------------------


    # veracity
    predictions = np.concatenate(predictions, axis=0)
    labels = np.array([eval_dataset.label_dict[l]  for l in eval_dataset.label])
    recall = recall_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")


    # save evidence details
    if args.get_evidences:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, dataset_type + '_'+ str(args.top_k) +'_evidence_details.json'), 'w') as f:
            json.dump(tmp_evidence_dict, f)

    results = {
                # "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "loss": eval_loss/(iter_count),

            }

    return results

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dataset_name", default="LIAR_RAW", type=str)

    parser.add_argument("--correlation_method", default="mlp", type=str)
    parser.add_argument("--cls_loss_weight", default=0.9, type=float,
                        help="The weight of the classification loss in the final loss. Range from [0, 1]. "
                             "The weight of the kl loss will be 1 - cls_loss_weight.")
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--report_each_claim", default=30, type=int)
    # parser.add_argument("--num_labels", default=1, type=int)


    parser.add_argument("--save_checkpoint",type=bool,default=False)

    # function
    parser.add_argument("--get_evidences", action='store_true')



    define_hparams_training(parser)
    args = parser.parse_args()
    set_seed(args)
    setup_logging(args)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device


    train_raw_dataset, eval_raw_dataset, test_raw_dataset = get_raw_datasets(args.dataset_name)

    config_class, tokenizer_class, model_class = MODEL_CLASSES[args.model_name]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # add config
    config.correlation_method = args.correlation_method
    # config.num_labels = args.num_labels
    config.cls_loss_weight = args.cls_loss_weight

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    train_dataset = NewsDataset(args.dataset_name, train_raw_dataset, tokenizer, args.max_seq_length, report_each_claim=args.report_each_claim)
    eval_dataset = NewsDataset(args.dataset_name, eval_raw_dataset, tokenizer, args.max_seq_length)
    test_dataset = NewsDataset(args.dataset_name, test_raw_dataset, tokenizer, args.max_seq_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=eval_dataset, eval_fn=evaluation)

    if args.do_prediction: # view the performance of the temp detection
        results = evaluation(args, test_dataset, model, tokenizer, global_step=None, mode='test')
        with open(os.path.join(args.output_dir, str(args.top_k) + '_test_result.txt'), 'w') as wf:
            for _i, (k, v) in enumerate(results.items()):
                wf.write(k + ': {}%\n'.format(np.around(100 * v, 2)))
                print(k + ': {}%'.format(np.around(100 * v, 2)))

    if args.get_evidences:  # used to get the extracted evidences, call it after the training.
        for dataset, dtype in zip([eval_dataset, test_dataset, train_dataset], ['eval', 'test', 'train']):
            results = evaluation(args, dataset, model, tokenizer, global_step=None, mode='test', dataset_type=dtype)




if __name__ == '__main__':
    main()

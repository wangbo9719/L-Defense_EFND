import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse

from help import *

from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer

import torch
from source.dataset import get_raw_datasets, Stage2PredictionDatasetForRoBERTa
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

MODEL_CLASSES = {
    "roberta": (RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer),
}


ID2LABEL_3cls = {0:"false", 1:"half", 2:"true"}
ID2LABEL_6cls = {0:"pants-fire", 1:"false", 2:"barely-true", 3:"half-true", 4:"mostly-true", 5:"true"}


def train(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=None):

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    eval_fn = eval_fn
    metric_best = -1e-5

    # --------------------- dataloader ---------------------
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=train_dataset.data_collate_fn)

    # --------------------- num of steps ---------------------
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.t_total = t_total

    # --------------------- optimizer and scheduler ---------------------
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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=adam_betas, eps=args.adam_epsilon,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # --------------------- training ---------------------
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Total train batch size = %d", args.t_total)

    ### training prepare
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])


    ### training
    for _idx_epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps), disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            ### forward
            model.train()
            inputs, inputs_text_list = tuple(t.to(args.device) for t in batch[0]), batch[1]
            if args.model_name == 't5':
                input_ids, labels = inputs[0], inputs[1]
                outputs = model(input_ids = input_ids, labels=labels)
            elif args.model_name == 'roberta' or args.model_name == 'distilbert':
                input_ids, attention_mask, labels = inputs[0], inputs[1], inputs[2]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                raise NotImplementedError

            ### loss
            loss = outputs.loss
            loss.backward()
            tr_loss += loss.item()

            ### logging
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0.001:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)

                global_step += 1

                # logging
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    logging.info(
                        f'Ep[{_idx_epoch}] Loss: {loss:.4f}')
                    with open(os.path.join(args.output_dir, 'loss_record.txt'), 'a') as wf:
                        wf.write('ep:' + str(_idx_epoch) + ', global_step:' + str(global_step) + ': ')
                        wf.write(f'Loss: {loss:.4f}')
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

        # do eval at the end of each epoch
        results = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
        with open(os.path.join(args.output_dir, 'eval_results.txt'), 'a') as wf:
            wf.write('End of ep' + str(_idx_epoch) + ', global_step:' + str(global_step) + ': \n')
            for key, value in results.items():
                wf.write(key + ':' + f'{100 * value:.2f}\n')
            wf.write('\n')
        if results['f1'] > metric_best:
            save_model_with_default_name(args.output_dir, model, tokenizer, args)
            metric_best = results['f1']



    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return metric_best

def evaluation(args, eval_dataset, model, tokenizer, global_step=0, type='train'):

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.data_collate_fn)

    ### logging
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))

    ### prepare
    model.eval()
    eval_loss = 0.0

    predictions = []
    labels = []

    ### eval

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs, inputs_text_list = tuple(t.to(args.device) for t in batch[0]), batch[1]
            input_ids, attention_masks, batch_labels = inputs[0], inputs[1], inputs[2]
            outputs = model(input_ids, attention_masks, labels=batch_labels)

        logits = outputs.logits
        prediction = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        predictions.append(prediction.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())

    # veracity
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    recall = recall_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    accuracy = accuracy_score(labels, predictions)

    results = {
        # "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        'accuracy': accuracy,
    }


    return results

def get_final_explanations(args, eval_dataset, model, tokenizer, global_step=0, type=''):

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.data_collate_fn)

    ### logging
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))

    ### prepare
    model.eval()
    eval_loss = 0.0

    predictions = []
    labels = []


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs, inputs_text_list = tuple(t.to(args.device) for t in batch[0]), batch[1]
            input_ids, attention_masks, batch_labels = inputs[0], inputs[1], inputs[2]
            outputs = model(input_ids, attention_masks, labels=batch_labels)

        logits = outputs.logits
        prediction = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        predictions.append(prediction.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())

    # veracity
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    recall = recall_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    accuracy = accuracy_score(labels, predictions)

    results = {
        # "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        'accuracy': accuracy,
    }

    if type == 'infer':
        final_results = []
        claims = eval_dataset.claims
        event_ids = eval_dataset.event_ids
        gold_explains = eval_dataset.explains
        explanations = eval_dataset.explanations

        if args.num_labels == 3:
            ID2LABEL = ID2LABEL_3cls
        else:
            ID2LABEL = ID2LABEL_6cls

        for i in range(len(eval_dataset)):
            sample = {}

            sample['label'] = ID2LABEL[predictions[i]]

            model_explanation = ""
            if ID2LABEL[predictions[i]] == 'true' or ID2LABEL[predictions[i]] == 'mostly-true':
                model_explanation = explanations[2 * i]
            elif ID2LABEL[predictions[i]] == 'false' or ID2LABEL[predictions[i]] == 'barely-true' or ID2LABEL[predictions[i]] == 'pants-fire':
                model_explanation = explanations[2 * i + 1]
            elif ID2LABEL[predictions[i]] == 'half' or ID2LABEL[predictions[i]] == 'half-true':
                model_explanation = "By combining the reasoning behind the false and true explanations, " \
                                    "we can discern the nuanced nature of this news, leading us to classify it as half true.\n"
                model_explanation += "True Explanation: " + explanations[2 * i] + "\n"
                model_explanation += "False Explanation: " + explanations[2 * i + 1]
            else:
                raise ValueError('Wrong prediction')
            sample['explanation'] = model_explanation

            sample['idx'] = i
            sample['claim'] = claims[i]
            sample['event_id'] = event_ids[i]
            sample['gold_label'] = ID2LABEL[eval_dataset.labels[i]]
            sample['gold_explain'] = gold_explains[i]
            sample['true_explain'] = explanations[2*i]
            sample['false_explain'] = explanations[2*i+1]

            final_results.append(sample)

        save_json(final_results, os.path.join(args.output_dir, "my_" + args.explanation_type +'.json'))


    return results

def get_final_corresponds_evidences(args, eval_dataset, model, tokenizer, global_step=0, type=''):

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.data_collate_fn)

    ### logging
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))

    ### prepare
    model.eval()
    eval_loss = 0.0

    predictions = []
    labels = []

    ### eval

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs, inputs_text_list = tuple(t.to(args.device) for t in batch[0]), batch[1]
            input_ids, attention_masks, batch_labels = inputs[0], inputs[1], inputs[2]
            outputs = model(input_ids, attention_masks, labels=batch_labels)

        logits = outputs.logits
        prediction = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        predictions.append(prediction.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())

    # veracity
    predictions = list(np.concatenate(predictions, axis=0))
    labels = list(np.concatenate(labels, axis=0))


    final_evidence_sentences = []
    for _i, _pre in enumerate(predictions):
        sample = {}
        sample['idx'] = _i
        sample['pred_label'] = ID2LABEL_3cls[_pre]
        sample['true_label'] = ID2LABEL_3cls[labels[_i]]
        if _pre == 0:
            sample['evidence'] = eval_dataset.false_sents[_i][:10]  # select top 10
        elif _pre == 2:
            sample['evidence'] = eval_dataset.true_sents[_i][:10]
        elif _pre == 1:
            sample['evidence'] = eval_dataset.false_sents[_i][:5] + eval_dataset.true_sents[_i][:5]
        else:
            raise ValueError('Wrong prediction')
        final_evidence_sentences.append(sample)

    with open(os.path.join(args.output_dir, "RAWFC_llama2_final_evidences.json"), 'w') as f:
        json.dump(final_evidence_sentences, f)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_class", default="roberta", type=str,
    #                     help="model class, one of [bert, roberta]")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--dataset_name", default="RAWFC_step2", type=str)
    parser.add_argument("--dataset_type", default="train", type=str)


    parser.add_argument("--model_name", default="roberta", type=str)

    parser.add_argument("--save_checkpoint", type=bool, default=False)


    parser.add_argument("--get_final_explanations", action='store_true')
    parser.add_argument("--do_filter", action='store_true')

    parser.add_argument("--explanation_type", default=None, type=str,
                        help="explanation type, one of [llama2, gpt]")
    parser.add_argument("--num_evidence_sentences", default=10, type=int,
                        help="The number of evidence sentences fed into the explanation generation module")


    # rebuttal
    parser.add_argument("--get_final_corresponds_evidences", action='store_true')



    define_hparams_training(parser)
    args = parser.parse_args()
    set_seed(args)
    setup_logging(args)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device



    train_raw_dataset, eval_raw_dataset, test_raw_dataset = get_raw_datasets(args.dataset_name)
    assert args.explanation_type is not None

    model_class, config_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = config_class.from_pretrained(args.model_name_or_path)
    args.num_labels = 3 if "RAWFC" in args.dataset_name else 6
    config.num_labels = args.num_labels
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_dataset = Stage2PredictionDatasetForRoBERTa(args.dataset_name, train_raw_dataset, tokenizer, 512, 'train',
                                                      explanation_type=args.explanation_type, nums_label=args.num_labels, do_filter=args.do_filter)
    eval_dataset = Stage2PredictionDatasetForRoBERTa(args.dataset_name, eval_raw_dataset, tokenizer, 512, 'eval',
                                                     explanation_type=args.explanation_type, nums_label=args.num_labels, do_filter=args.do_filter)
    test_dataset = Stage2PredictionDatasetForRoBERTa(args.dataset_name, test_raw_dataset, tokenizer, 512, 'test',
                                                     explanation_type=args.explanation_type, nums_label=args.num_labels, do_filter=args.do_filter)

    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=eval_dataset, eval_fn=evaluation)
    if args.do_prediction:
        results = evaluation(args, test_dataset, model, tokenizer, type='infer')
        with open(os.path.join(args.output_dir, 'test_result.txt'), 'w') as wf:
            for _i, (k, v) in enumerate(results.items()):
                wf.write(k + ': {}%\n'.format(np.around(100 * v, 2)))
                print(k + ': {}%\n'.format(np.around(100 * v, 2)))

    if args.get_final_explanations:
        get_final_explanations(args, test_dataset, model, tokenizer, type='infer')
        print('get final explanations done')

    if args.get_final_corresponds_evidences:
        get_final_corresponds_evidences(args, test_dataset, model, tokenizer, type='infer')
        print('get final corresponds explanations done')


if __name__ == '__main__':


    main()



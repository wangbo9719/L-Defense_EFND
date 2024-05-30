import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from torch.nn.utils.rnn import pad_sequence

def minmax_norm(arr, a=0., b=1.0):
    min_val, max_val = torch.min(arr), torch.max(arr)
    return (arr - min_val) * ((b-a) / (max_val - min_val)) + a


class EvidenceSelection(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        # For TRUE attention
        self.fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, 1)
        # For False attention
        self.fc3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.fc4 = nn.Linear(config.hidden_size, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.correlation_method = config.correlation_method
        self.corre_fc1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.corre_fc2 = nn.Linear(config.hidden_size, 2)

        self.cls_loss_weight = config.cls_loss_weight

        self.predict_fc0 = nn.Linear(2, 3)
        self.predict_fc1 = nn.Linear(2 + config.hidden_size, 100)
        self.predict_fc2 = nn.Linear(100, 3)

    def true_attn(self, claim_embds, sents_embds):
        concat_embds = torch.cat([claim_embds.repeat(sents_embds.size(0), 1), sents_embds], dim=-1)
        x = self.fc1(concat_embds)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = minmax_norm(x)

        x = torch.softmax(x, dim=0).squeeze(1)
        return x

    def false_attn(self, claim_embds, sents_embds):
        concat_embds = torch.cat([claim_embds.repeat(sents_embds.size(0), 1), sents_embds], dim=-1)
        x = self.fc3(concat_embds)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = minmax_norm(x)

        x = torch.softmax(x, dim=0).squeeze(1)
        return x


    def classifier(self, claim_embds, scores):
        concat_embds = torch.cat([claim_embds, scores], dim=-1)
        x = self.predict_fc1(concat_embds)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.predict_fc2(x)
        x = torch.softmax(x, dim=-1)
        return x

    def get_correlation_scores(self, claim_embds, sents_embds):

        sample_corre = None
        if self.correlation_method == 'mlp':
            concat_embds = torch.cat([claim_embds * sents_embds, claim_embds-sents_embds, claim_embds.repeat(sents_embds.size(0), 1), sents_embds], dim=-1)
            x = self.corre_fc1(concat_embds)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.corre_fc2(x).squeeze()

            sample_corre = torch.softmax(x, dim=-1)

        return sample_corre


    def encoder(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)    # CHECK
        pooled_output = outputs[1]  # [CLS]
        return pooled_output

    def forward(self, claim_ids, claim_mask_ids, claim_labels,
                sents_ids, sents_mask_ids, sents_labels, num_sentence_per_report):

        bs, num_sents, seq_len = sents_ids.shape
        claim_embds = self.encoder(claim_ids, claim_mask_ids) # [bs, hidden_size]
        batch_num_sents = [sum(torch.where(sents_labels != -1)[0] == _i) for _i in range(bs)]
        segment_indexes = [sum(batch_num_sents[:_i]) for _i in range(bs+1)]
        valid_sent_ids = torch.concat([sents_ids[_i, :num_sents] for _i, num_sents in enumerate(batch_num_sents)])
        valid_sent_mask_ids = torch.concat([sents_mask_ids[_i, :num_sents] for _i, num_sents in enumerate(batch_num_sents)])
        sents_embds = self.encoder(valid_sent_ids, valid_sent_mask_ids) # [batch_total_num_sents, hidden_size] ?

        true_scores, true_attn_weights = [], []
        false_scores, false_attn_weights = [], []

        target_soft_labels = []
        target_hard_labels = []

        for _i in range(bs):
            sample_corre = self.get_correlation_scores(claim_embds[_i], sents_embds[segment_indexes[_i]:segment_indexes[_i + 1]]).view(-1, 2) # [batch_num_sents]
            sample_true_attn = self.true_attn(claim_embds[_i], sents_embds[segment_indexes[_i]:segment_indexes[_i + 1]]) # [batch_num_sents]
            sample_false_attn = self.false_attn(claim_embds[_i], sents_embds[segment_indexes[_i]:segment_indexes[_i + 1]]) # [batch_num_sents]

            if sample_corre.shape[0] == 1:
                sents_true_score = sample_corre[:, 1]
                sents_false_score = sample_corre[:, 0]
            else:
                sample_true_attn = self.true_attn(claim_embds[_i], sents_embds[segment_indexes[_i]:segment_indexes[_i + 1]])  # [batch_num_sents]
                sample_false_attn = self.false_attn(claim_embds[_i], sents_embds[segment_indexes[_i]:segment_indexes[_i + 1]])  # [batch_num_sents]

                sents_true_score = sample_true_attn * sample_corre[:, 1]   # [batch_num_sents]
                sents_false_score = sample_false_attn * sample_corre[:, 0] # [batch_num_sents]

            # collection
            true_scores.append(sents_true_score)
            false_scores.append(sents_false_score)

            true_attn_weights.append(sample_true_attn)
            false_attn_weights.append(sample_false_attn)

            # target_soft_labels for kl-loss
            if claim_labels[_i] == 0:
                target_soft_labels.append(torch.tensor([1.0, 0.0]).to(claim_labels.device))
                target_hard_labels.append(torch.tensor([1.0, 0.0, 0.0]).to(claim_labels.device))
            elif claim_labels[_i] == 1:
                target_soft_labels.append(torch.tensor([0.5, 0.5]).to(claim_labels.device))
                target_hard_labels.append(torch.tensor([0.0, 1.0, 0.0]).to(claim_labels.device))
            elif claim_labels[_i] == 2:
                target_soft_labels.append(torch.tensor([0.0, 1.0]).to(claim_labels.device))
                target_hard_labels.append(torch.tensor([0.0, 0.0, 1.0]).to(claim_labels.device))
            else:
                raise ValueError


        padded_true_scores = pad_sequence(true_scores, batch_first=True, padding_value=0)
        padded_false_scores = pad_sequence(false_scores, batch_first=True, padding_value=0)

        sample_true_attned_probs = torch.sum(padded_true_scores, dim=1)
        sample_false_attned_probs = torch.sum(padded_false_scores, dim=1)
        predict_scores = torch.cat([sample_false_attned_probs.unsqueeze(1), sample_true_attned_probs.unsqueeze(1)], dim=1)


        target_soft_labels = torch.stack(target_soft_labels, dim=0)
        kl_loss_fn = nn.KLDivLoss(reduction='mean')
        kl_loss = kl_loss_fn(torch.log_softmax(predict_scores, dim=-1), target_soft_labels)

        # cls loss
        cls_logits = self.classifier(claim_embds, predict_scores)
        cls_loss_fn = nn.CrossEntropyLoss()
        target_hard_labels = torch.stack(target_hard_labels, dim=0)
        cls_loss = cls_loss_fn(cls_logits, target_hard_labels)


        loss = (1-self.cls_loss_weight) * kl_loss + self.cls_loss_weight * cls_loss

        return loss, kl_loss, cls_loss, cls_logits, predict_scores, true_scores, false_scores, true_attn_weights, false_attn_weights


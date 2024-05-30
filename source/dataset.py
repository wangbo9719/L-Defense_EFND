from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join as pjoin
from help import *


HOME_DIR = BASE_DIR + "/EFND_L-Defense/"


DATASET2PATH = {
    "LIAR_RAW": os.path.join(HOME_DIR + "dataset/LIAR-RAW/"),
    "RAWFC": os.path.join(HOME_DIR + "dataset/RAWFC/"),

    "RAWFC_step2": os.path.join(HOME_DIR + "dataset/RAWFC_step2/"),
    "LIAR-RAW_step2": os.path.join(HOME_DIR + "dataset/LIAR-RAW_step2/"),

}

LABEL_IDS = {
    "LIAR_RAW":{"pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 1, "mostly-true": 2, "true": 2},
    "RAWFC":{"false": 0, "half": 1, "true": 2},
    "LIAR_RAW_SIX":{"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5},
}

def get_LIAR_six_cls_labels(dataset_type):
    train_dataset_raw, dev_dataset_raw, test_dataset_raw = get_raw_datasets("LIAR_RAW")
    labels = {}
    if dataset_type == 'train':
        dataset = train_dataset_raw
    elif dataset_type == 'eval':
        dataset = dev_dataset_raw
    elif dataset_type == 'test':
        dataset = test_dataset_raw

    for obj in tqdm(dataset):
        # each sample has one
        event_id, label = obj['event_id'], obj['label']
        labels[event_id] = LABEL_IDS["LIAR_RAW_SIX"][label]

    return labels


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        news_dataset = json.load(f)  # a list contain all data.
    return news_dataset

def read_RAWFC(path):
    filenames = os.listdir(path)
    name_list = []
    for name in filenames:
        if '.json' in name:
            name_list.append(name)

    if len(name_list) == 1:
        all_data = ''
        for file in name_list:
            filename = pjoin(path, file)
            with open(filename, 'r', encoding='utf-8') as json_file:
                all_data = json.load(json_file)
    else:
        all_data = []
        for file in name_list:
            filename = pjoin(path, file)
            with open(filename, 'r', encoding='utf-8') as json_file:
                obj = json.load(json_file)
                all_data.append(obj)
    return all_data

def get_raw_datasets(dataset, dataset_dir=None):
    if dataset == "LIAR_RAW":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train.json")
        dev_dataset_path = os.path.join(dataset_dir, "val.json")
        test_dataset_path = os.path.join(dataset_dir, "test.json")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset == 'RAWFC':
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train")
        dev_dataset_path = os.path.join(dataset_dir, "val")
        test_dataset_path = os.path.join(dataset_dir, "test")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_RAWFC(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset == "RAWFC_step2" or dataset == "LIAR_RAW_step2":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train_10_evidence_details.json")  # if top-k == 10
        dev_dataset_path = os.path.join(dataset_dir, "eval_10_evidence_details.json")
        test_dataset_path = os.path.join(dataset_dir, "test_10_evidence_details.json")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]

    else:
        raise NotImplementedError(dataset, dataset_dir)
    return train_dataset_raw, dev_dataset_raw, test_dataset_raw

# Used in Step-1
class NewsDataset(Dataset):
    def __init__(
            self, dataset_name, news_dataset, tokenizer, max_seq_length=128,
            nums_label=6, report_each_claim=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.nums_label = nums_label
        self.label_dict = LABEL_IDS[dataset_name]
        self.dataset_path = None
        self.report_each_claim = report_each_claim

        self.example_list = news_dataset
        self.event_id, self.claim, self.label, self.explain, \
            self.report_links, self.report_contents, self.report_domains, \
            self.report_sents, self.report_sents_labels, \
            self.report_sents_belong, self.num_sentences_per_report = self.load_raw(news_dataset)

        self._sep_id, self._cls_id, self._pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token, self.tokenizer.cls_token, self.tokenizer.pad_token, ]
        )

    def load_raw(self, dataset):
        '''parsing dict objs to list '''
        # event_id, claim, label, explain, (link, content, domain, report_sents, report_sents_is_evidence, report_sents_belong_which_report)
        raw_data = [[] for _ in range(11)]
        for obj in tqdm(dataset):
            # each sample has one
            event_id, claim, label, explain, reports = \
                obj['event_id'], obj['claim'], obj['label'], obj['explain'], obj['reports']
            raw_data[0].append(event_id)
            raw_data[1].append(claim)
            raw_data[2].append(label)
            raw_data[3].append(explain)

            # each sample has many reports
            report_links = []
            report_contents = []
            report_domains = []

            report_sents = []
            report_sents_labels = []
            report_sents_belong = []

            num_sentences_per_report = []

            for r_id, report in enumerate(reports[:self.report_each_claim]):  # clip
                report_links.append(report['link'])
                report_contents.append(report['content'])  # complete report
                report_domains.append(report['domain'])
                num_sentences_per_report.append(len(report['tokenized']))
                # each report has many sentences
                for sentence in report['tokenized']:
                    report_sents.append(sentence['sent'])
                    report_sents_labels.append(sentence['is_evidence'])
                    report_sents_belong.append(r_id)

            raw_data[4].append(report_links)
            raw_data[5].append(report_contents)
            raw_data[6].append(report_domains)

            raw_data[7].append(report_sents)
            raw_data[8].append(report_sents_labels)
            raw_data[9].append(report_sents_belong)

            raw_data[10].append(num_sentences_per_report)

        return raw_data

    def __getitem__(self, index):

        claim_tokenized = self.tokenizer(self.claim[index], return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_length)
        claim_input_ids, claim_mask_ids = claim_tokenized['input_ids'].squeeze(), claim_tokenized['attention_mask'].squeeze()
        claim_label_id = torch.tensor(self.label_dict[self.label[index]], dtype=torch.long)

        sent_tokenized = self.tokenizer(self.report_sents[index], return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_length)
        sent_input_ids, sent_mask_ids = sent_tokenized['input_ids'], sent_tokenized['attention_mask']
        sent_label_ids = torch.tensor(self.report_sents_labels[index], dtype=torch.long)

        decoder_input_tokenized = self.tokenizer(self.explain[index], return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_length)
        decoder_input_ids, decoder_mask_ids = decoder_input_tokenized['input_ids'].squeeze(), decoder_input_tokenized['attention_mask'].squeeze()

        num_sentences_per_report = torch.tensor(self.num_sentences_per_report[index], dtype=torch.long)

        raw_text_dict = {}
        raw_text_dict['event_id'] = self.event_id[index]
        raw_text_dict['claim'] = self.claim[index]
        raw_text_dict['sents'] = self.report_sents[index]
        raw_text_dict['sents_labels'] = self.report_sents_labels[index]
        raw_text_dict['explain'] = self.explain[index]

        return claim_input_ids, claim_mask_ids, claim_label_id, \
               sent_input_ids, sent_mask_ids, sent_label_ids, \
               decoder_input_ids, decoder_mask_ids, \
               num_sentences_per_report, raw_text_dict

    def data_collate_fn(self, batch):

        raw_data_list = list(zip(*batch))
        tensors_list, raw_text_list = raw_data_list[:-1], raw_data_list[-1]

        return_list = []
        # PADDING
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t % 3 == 0:
                padding_value = self._pad_id
            elif _idx_t == 5:
                padding_value = -1  # padding for sent_labels
            else:
                padding_value = 0

            if _idx_t == 3 or _idx_t == 4:  # sent_input_ids, sent_mask_ids
                # 2D padding
                _max_len_last_dim = 0
                for _tensor in _tensors:
                    _local_max_len_last_dim = max(len(_t) for _t in list(_tensor))
                    _max_len_last_dim = max(_max_len_last_dim, _local_max_len_last_dim)
                # padding
                _new_tensors = []
                for _tensor in _tensors:
                    inner_tensors = []
                    for idx, _ in enumerate(list(_tensor)):
                        _pad_shape = _max_len_last_dim - len(_tensor[idx])
                        _pad_tensor = torch.tensor([padding_value] * _pad_shape, device=_tensor[idx].device, dtype=_tensor[idx].dtype)
                        _new_inner_tensor = torch.cat([_tensor[idx], _pad_tensor], dim=0)
                        inner_tensors.append(_new_inner_tensor)
                    _tensors_tuple = tuple(ts for ts in inner_tensors)
                    _new_tensors.append(torch.stack(_tensors_tuple, dim=0))
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(_new_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                if _tensors[0].dim() >= 1:
                    return_list.append(
                        torch.nn.utils.rnn.pad_sequence(_tensors, batch_first=True, padding_value=padding_value),
                    )
                else:
                    return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list), raw_text_list

    def __len__(self):
        return len(self.example_list)


# Used in Step-2: Prepare the prompt based on the extracted sentences.
class Stage2DatasetForLLM(Dataset):
    def __init__(
            self, dataset_name, news_dataset,
            nums_label=3, num_evidence_sentences=10, *args, **kwargs):

        self.dataset_name = dataset_name
        self.nums_label = nums_label
        self.num_evidence_sentences = num_evidence_sentences

        self.example_list = news_dataset
        self.event_ids, self.claims, self.labels, self.explains, \
            self.true_sents, self.true_scores, self.true_idx, \
            self.false_sents, self.false_scores, self.false_idx = self.load_raw(news_dataset)

        self.true_evidences, self.false_evidences = self.format(self.true_sents, self.false_sents)


    def load_raw(self, news_dataset):
        raw_data = [[] for _ in range(10)]

        for event_id, details in news_dataset.items():
            raw_data[0].append(event_id)
            raw_data[1].append(details['claim'])
            raw_data[2].append(details['label'])
            raw_data[3].append(details['explain'])
            true_sents, true_scores, true_idx, false_sents, false_scores, false_idx = [], [], [], [], [], []
            for sent_details in details['true_details']:
                true_sents.append(sent_details[0])
                true_scores.append(sent_details[1])
                true_idx.append(sent_details[2])
            for sent_details in details['false_details']:
                false_sents.append(sent_details[0])
                false_scores.append(sent_details[1])
                false_idx.append(sent_details[2])
            raw_data[4].append(true_sents)
            raw_data[5].append(true_scores)
            raw_data[6].append(true_idx)
            raw_data[7].append(false_sents)
            raw_data[8].append(false_scores)
            raw_data[9].append(false_idx)
        return raw_data


    def format(self, true_sents, false_sents):
        true_evidences, false_evidences = [], []
        for idx in range(len(true_sents)):

            true_evidences_list = [sent for i, sent in enumerate(true_sents[idx]) if
                                   i < self.num_evidence_sentences]
            sample_true_evidences = '\n'.join(true_evidences_list)
            false_evidences_list = [sent for i, sent in enumerate(false_sents[idx]) if
                                    i < self.num_evidence_sentences]
            sample_false_evidences = '\n'.join(false_evidences_list)

            true_evidences.append(sample_true_evidences)
            false_evidences.append(sample_false_evidences)


        return true_evidences, false_evidences

    def __getitem__(self, index):

        return self.claims[index], self.true_evidences[index], self.false_evidences[index]

    def __len__(self):
        return len(self.claims)


# Used in Step-3
class Stage2PredictionDatasetForRoBERTa(Dataset):
    def __init__(
            self, dataset_name, news_dataset, tokenizer, max_seq_length,
            dataset_type, nums_label=6, explanation_type=None, do_filter=False, *args, **kwargs):

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


        self.nums_label = nums_label

        self.example_list = news_dataset
        self.event_ids, self.claims, self.labels, self.explains, \
            self.true_sents, self.true_scores, self.true_idx, \
            self.false_sents, self.false_scores, self.false_idx = self.load_raw(news_dataset)

        self.explanations = load_json(os.path.join(HOME_DIR + "dataset/" + self.dataset_name + "/" + explanation_type + "/" + dataset_type + "_label_oriented_explanation.json"))
        self.do_filter = do_filter

        self.prompt = 'claim: [CLAIM] [SEP] true-oriented explanation: [TRUE_EXPLANATION] [SEP] false-oriented explanation: [FALSE_EXPLANATION]'


        self.inputs_list = self.format()

    def load_raw(self, news_dataset):
        raw_data = [[] for _ in range(10)]
        if self.nums_label == 6:
            assert "LIAR" in self.dataset_name, "Invalid dataset_name for six labels: {}".format(self.dataset_name)
            six_cls_labels = get_LIAR_six_cls_labels(self.dataset_type)
        for event_id, details in news_dataset.items():
            raw_data[0].append(event_id)
            raw_data[1].append(details['claim'])
            if self.nums_label == 3:
                raw_data[2].append(details['label'])
            elif self.nums_label == 6:
                raw_data[2].append(six_cls_labels[event_id])
            else:
                raise ValueError('Invalid nums_label: {}'.format(self.nums_label))
            raw_data[3].append(details['explain'])
            true_sents, true_scores, true_idx, false_sents, false_scores, false_idx = [], [], [], [], [], []
            count = 0
            for sent_details in details['true_details']:
                true_sents.append(sent_details[0])
                true_scores.append(sent_details[1])
                true_idx.append(sent_details[2])
                count += 1
                if count >= 20:
                    break
            count = 0
            for sent_details in details['false_details']:
                false_sents.append(sent_details[0])
                false_scores.append(sent_details[1])
                false_idx.append(sent_details[2])
                count += 1
                if count >= 20:
                    break
            raw_data[4].append(true_sents)
            raw_data[5].append(true_scores)
            raw_data[6].append(true_idx)
            raw_data[7].append(false_sents)
            raw_data[8].append(false_scores)
            raw_data[9].append(false_idx)
        return raw_data

    def format(self):
        explanations = self.explanations
        inputs_list = []
        for i, claim in tqdm(enumerate(self.claims)):
            true_exp = explanations[2 * i]
            false_exp = explanations[2 * i + 1]
            if self.do_filter:
                if 'The claim' in true_exp[:20]:
                    first_sent = true_exp.split('. ')[0]
                    true_exp = true_exp[len(first_sent) + 2:]
                if 'The claim' in false_exp[:20]:
                    first_sent = false_exp.split('. ')[0]
                    false_exp = false_exp[len(first_sent) + 2:]
            input = self.prompt.replace("[CLAIM]", claim).replace("[TRUE_EXPLANATION]", "[" + true_exp + "]").replace(
                "[FALSE_EXPLANATION]", "[" + false_exp + "]")
            inputs_list.append(input)

        return inputs_list

    def __getitem__(self, index):

        sent = self.inputs_list[index]
        input_id, _ = self.tokenizer(sent, return_tensors='pt', truncation=True, max_length=self.max_seq_length).values()
        attention_mask = torch.ones_like(input_id)


        label = torch.tensor(self.labels[index], dtype=torch.long)

        return self.inputs_list[index], input_id.squeeze(0), attention_mask.squeeze(0), label

    def data_collate_fn(self, batch):

        raw_data_list = list(zip(*batch))
        inputs_text_list, tensors_list = raw_data_list[0], raw_data_list[1:]

        return_list = []
        # PADDING
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t == 0:  # input_ids
                padding_value = self.tokenizer.pad_token_id  # padding for input_ids
            elif _idx_t == 1:  # attention_mask
                padding_value = 0  # padding for attention_mask

            # if _idx_t == 0:  # sent_input_ids, sent_mask_ids
            if _tensors[0].dim() >= 1:
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list), inputs_text_list

    def __len__(self):
        return len(self.inputs_list)










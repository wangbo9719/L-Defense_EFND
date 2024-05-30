import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse

from help import *

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from source.dataset import get_raw_datasets, Stage2DatasetForLLM, DATASET2PATH
from tqdm import tqdm, trange

from prompts import *


DATASET_DIR = BASE_DIR + "/EFND_L-Defense/dataset"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = B_SYS + EXPLANATION_SYSTEM_PROMPT_v0 + E_SYS


ID2LABEL_3cls = {0:"false", 1:"half", 2:"true"}
ID2LABEL_6cls = {0:"pants-fire", 1:"false", 2:"barely-true", 3:"half-true", 4:"mostly-true", 5:"true"}


def get_prompt(instructions):
    prompts = []
    for ins in instructions:
        prompt = B_INST + SYSTEM_PROMPT + ins + E_INST
        prompts.append(prompt)
    return prompts

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index == -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")

def get_reference_explanation_llama2(args, dataset, dataset_type):
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map = "auto",
                                                 torch_dtype=torch.float16,)

    results = []
    count = 0
    prompts = get_prompts_for_explanation_generation(dataset.claims, dataset.true_evidences, dataset.false_evidences, tokenizer)

    save_dir = args.output_dir + "/llama2/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for _idx, prompt in enumerate(tqdm(prompts, desc=dataset_type + '_llama2 explanation generation')):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs,

                              do_sample=True,
                              top_k=10,
                              max_new_tokens=512,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.eos_token_id)
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')  # should output a list
        final_outputs = remove_substring(final_outputs, prompt)

        res_dict = dict()
        res_dict['idx'] = _idx
        res_dict['explanation'] = final_outputs
        results.append(res_dict)
        count += 1
        if count == 100:
            count = 0
            save_json(results, save_dir + dataset_type + "_label_oriented_explanation.json")

    save_json(results, save_dir + dataset_type + "_label_oriented_explanation.json")

def get_reference_explanation_gpt(args, dataset, dataset_type):

    prompts = get_prompts_for_explanation_generation(dataset.claims, dataset.true_evidences, dataset.false_evidences)

    results = []
    count = 0

    save_dir = args.output_dir + "gpt/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for _idx, prompt in enumerate(tqdm(prompts, desc=dataset_type + '_gpt_turbo explanation generation')):
        completion = client.chat.completions.create(
            # model="gpt-4",
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT_v0},
                {"role": "user", "content": prompt},
            ]
        )

        res = completion.choices[0].message
        # print(res)
        res_dict = dict()
        res_dict['idx'] = _idx
        res_dict['explanation'] = res.content
        results.append(res_dict)
        count += 1

    save_json(results, save_dir + dataset_type + "_label_oriented_explanation.json")

def get_prompts_for_explanation_generation(claims, true_sentences, false_sentences, tokenizer=None):
    prompts = []
    if tokenizer is not None: # for llama2

        for i, claim in tqdm(enumerate(claims), desc='Generate context'):
            true_instruction = EXPLANATION_USER_PROMPT_v0.replace("[CLAIM]", claim) \
                .replace("[LABEL]", "True") \
                .replace("[SENTENCES]", true_sentences[i])
            false_instruction = EXPLANATION_USER_PROMPT_v0.replace("[CLAIM]", claim) \
                .replace("[LABEL]", "False") \
                .replace("[SENTENCES]", false_sentences[i])

            true_instruction = B_INST + SYSTEM_PROMPT + true_instruction + E_INST
            false_instruction = B_INST + SYSTEM_PROMPT + false_instruction + E_INST
            prompts.append(true_instruction)
            prompts.append(false_instruction)

    else: # For GPT-turbo
        for i, claim in tqdm(enumerate(claims), desc='Generate context'):
            true_instruction = EXPLANATION_USER_PROMPT_v0.replace("[CLAIM]", claim)\
                                                    .replace("[LABEL]", "True")\
                                                    .replace("[SENTENCES]", true_sentences[i])
            false_instruction = EXPLANATION_USER_PROMPT_v0.replace("[CLAIM]", claim)\
                                                    .replace("[LABEL]", "False")\
                                                    .replace("[SENTENCES]", false_sentences[i])
            prompts.append(true_instruction)
            prompts.append(false_instruction)

    return prompts

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="RAWFC_step2", type=str)
    parser.add_argument("--dataset_type", default="all", type=str,
                        help="one of {train, eval, test, all}")

    parser.add_argument("--generate_explanation_llama2", action='store_true')
    parser.add_argument("--generate_explanation_gpt", action='store_true')

    parser.add_argument("--num_evidence_sentences", default=10, type=int,
                        help="The number of evidence sentences fed into the explanation generation module")


    define_hparams_training(parser)
    args = parser.parse_args()

    # the dir to save the generated explanations
    args.output_dir = DATASET2PATH[args.dataset_name]


    set_seed(args)
    setup_logging(args)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device


    #  -------------------------- datasets --------------------------
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_raw_dataset, eval_raw_dataset, test_raw_dataset = get_raw_datasets(args.dataset_name)

    train_dataset = Stage2DatasetForLLM(args.dataset_name, train_raw_dataset, num_evidence_sentences=args.num_evidence_sentences)  # Stage2Dataset
    eval_dataset = Stage2DatasetForLLM(args.dataset_name, eval_raw_dataset, num_evidence_sentences=args.num_evidence_sentences)
    test_dataset = Stage2DatasetForLLM(args.dataset_name, test_raw_dataset, num_evidence_sentences=args.num_evidence_sentences)

    # --------------------------- get explanation ---------------------------
    get_explanation_func = None

    if args.generate_explanation_llama2:
        get_explanation_func = get_reference_explanation_llama2
    elif args.generate_explanation_gpt:
        get_explanation_func = get_reference_explanation_gpt

    if args.dataset_type == "train":
        get_explanation_func(args, train_dataset, args.dataset_type)
    elif args.dataset_type == "eval":
        get_explanation_func(args, eval_dataset, args.dataset_type)
    elif args.dataset_type == "test":
        get_explanation_func(args, test_dataset, args.dataset_type)
    elif args.dataset_type == "all":
        get_explanation_func(args, eval_dataset, "eval")
        get_explanation_func(args, test_dataset, "test")
        get_explanation_func(args, train_dataset, "train")
    else:
        raise ValueError('Wrong dataset type')


if __name__ == '__main__':



    main()



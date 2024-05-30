
This repo contains the source code of the paper accepted by WWW'2024 - 
[**"Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom"**](https://arxiv.org/pdf/2405.03371)

If possible, could you please star this project. ⭐ ↗️

## 1. Installing requirement packages
```
conda create -n efnd python=3.8
source activate efnd
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers pandas tqdm nltk sklearn tensorboardX openai
```


## 2. Running steps and commands

### 2.1 Train the extractor and get extracted top-k evidences

Train the extractor
```
CUDA_VISIBLE_DEVICES=0 python source/extractor_train.py \
--model_name_or_path roberta-base \   # roberta-large
--dataset_name RAWFC \
--output_dir ./runnings/step1_extraction_model_RAWFC \
--do_train \
--train_batch_size 2 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--eval_steps 500 \
--max_seq_len 64 \
--gradient_accumulation_steps 1 \
--correlation_method mlp \
--learning_rate 1e-5 \
--cls_loss_weight 0.9
```

Get extracted top-k evidences
```
CUDA_VISIBLE_DEVICES=0 python source/extractor_train.py \
--model_name_or_path ./runnings/step1_extraction_model_RAWFC \
--dataset_name RAWFC \
--output_dir ./dataset/RAWFC_step2 \
--get_evidences \
--train_batch_size 2 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--eval_steps 500 \
--max_seq_len 64 \
--gradient_accumulation_steps 1 \
--correlation_method mlp \
--learning_rate 1e-5 \
--cls_loss_weight 0.9
```

## 2.2 Get LLM-generated label oriented explanations
```
CUDA_VISIBLE_DEVICES=0 python ./source/step2_explanation_generation.py \
--dataset_name RAWFC_step2 \
--output_dir ./dataset/RAWFC_step2 \
--generate_explanation_gpt \
```

## 2.3 Final Prediction
```
CUDA_VISIBLE_DEVICES=0 python source/step3_final.py \
--dataset_name RAWFC_step2 \
--explanation_type gpt \
--model_name_or_path roberta-large \
--output_dir ./runnings/step3_RAWFC \
--do_train \
--do_prediction \
--train_batch_size 8 \
--num_train_epochs 5 \
--eval_batch_size 32 \
--eval_steps 500 \
--learning_rate 5e-6 \
```
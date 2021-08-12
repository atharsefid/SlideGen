# SlideGen: An Abstractive Section-Based Slide Generator for Scholarly Documents

This repository is the code for the paper entitled as "SlideGen: An Abstractive Section-Based Slide Generator for
Scholarly Documents"

## Data Preparation

### Step 1: Download the processed data

[Pre-processed data](https://drive.google.com/file/d/1xYHXYoQBa7DJVrq0ePly58ioq2EmmVG8/view)

Put all files into `raw_data` directory

### Step 2: Match sections to slide:

```
python3 match_slide_section.py
```

### Step 3: Generate Data files:

```
python3 data_generator/data_generator_utils.py 
```

# Model training

### Step 4: Start fine-tuning:

To train from the base model:

```
python3 Bart/fine_tune_bart.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --output_dir ./temp/bart --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --overwrite_output_dir --predict_with_generate --num_train_epochs 2
```

To train from a checkpoint:

```
python3 Bart/fine_tune_bart.py 
--model_name_or_path \
temp/t5_small/checkpoint-94500/ \
--do_train \
--do_eval \
--do_predict \
--output_dir \
./temp/t5_small \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--overwrite_output_dir \
--predict_with_generate \
--num_train_epochs \
5
```

### Test

```
python3 Bart/fine_tune_bart.py --model_name_or_path temp/bart/checkpoint-1000 --do_predict --output_dir ./temp/tst-summarization --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --overwrite_output_dir --predict_with_generate
```

### Generate final scores

```
python3 calculate_rouge_score.py 
```

<!-- 
experiments:
temp/tst-summarization3/ -> match with 256 tokens and trained with emtpy summaries
temp/tst-summarization1/ -> match with 128 tokens and trained without emtpy summaries
-->




# SlideGen: An Abstractive Section-Based Slide Generator for Scholarly Documents
This repository is the code for the paper entitled as "SlideGen: An Abstractive Section-Based Slide Generator for Scholarly Documents"


## Data Preparation 
### Step 1: download the processed data

[Pre-processed data](https://drive.google.com/file/d/1xYHXYoQBa7DJVrq0ePly58ioq2EmmVG8/view)

Put all files into `raw_data` directory


### Step 2: match sections to slide:

```
python3 match_slide_section.py
```

### Step 3: Make data suitable for BART fine-tuning:
Use data_generator/data_generator_utils.py to make data suitable for BART fine-tuning.

`generate_data_section_slide()` is responsible for creating json files for training by Bart.

### Start fine-tuning:
```
source venv/bin/activate

python3 fine_tune_bart.py \
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
<!-- 
experiments:
temp/tst-summarization3/ -> match with 256 tokens and trained with emtpy summaries
temp/tst-summarization1/ -> match with 128 tokens and trained without emtpy summaries
-->




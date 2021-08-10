# generate text with reference to a model
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

# t5-small \
# temp/tst-summarization2/checkpoint-19000 \
#facebook/bart-base \
#allenai/longformer-base-4096 \
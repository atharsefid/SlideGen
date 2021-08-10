python3 fine_tune_bart.py \
--model_name_or_path \
train_temp/tst-summarization/checkpoint-76000/ \
--do_predict \
--output_dir \
./temp/tst-summarization \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--overwrite_output_dir \
--predict_with_generate \

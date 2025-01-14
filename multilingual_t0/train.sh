python main.py \
	--model_name_or_path "google/mt5-small" \
	--seqio_mixture_name "sglue_train" \
	--do_train \
	--preprocessing_num_workers 4 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation 1\
	--overwrite_output_dir \
	--output_dir checkpoints/ \
	--max_steps 25000 \

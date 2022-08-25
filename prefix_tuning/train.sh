DATA_DIR='data_train/robust-prefix-tuning-data-models/data'
MODEL_ROOT_DIR='data_train/robust-prefix-tuning-data-models/saved_models'
python3 train.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"

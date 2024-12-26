# with DTR
DATASET_DIR='path/to/data_folder'  # path to the dataset
DATA_TYPE='ars548'  # oculii   ars548
STR_DIR="${DATASET_DIR}/7n5s_xy11_remove_${DATA_TYPE}"

echo $DATASET_DIR
echo $STR_DIR
echo $DATA_TYPE

python config_snail.py --remove --dataset_dir=$DATASET_DIR --data_type=$DATA_TYPE

python convert_snail.py --split='train' --struct_dir=$STR_DIR
python convert_snail.py --split='valid' --struct_dir=$STR_DIR
python convert_snail.py --split='test' --struct_dir=$STR_DIR

python ../generating_queries/generate_trainset_snail.py --struct_dir=$STR_DIR --data_type=$DATA_TYPE

python ../generating_queries/generate_testsets_snail.py --struct_dir=$STR_DIR --data_type=$DATA_TYPE
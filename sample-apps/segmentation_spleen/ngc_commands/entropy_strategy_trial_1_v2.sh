#!/bin/bash
NGC_CONTAINER="nvidian/pytorch:21.06-py3"
EXPORT_PATH="/workspace/al_code/MONAILabel"
FOLDER_PATH="/workspace/al_code/MONAILabel/sample-apps/segmentation_spleen"
NGC_WORKSPACE="mednet_pretrain:/workspace"
BASE_DIR="/workspace/al_results/entropy/trial_1_v2"
JSON_PATH="/workspace/al_code/spleen_json/dataset_0_init.json"
NGC_DATASET=69970
NGC_INSTANCE=dgx1v.16g.1.norm
EXP_NAME="AL_Entropy_Trial_1_v2"
BATCH_SIZE=4
VAL_BATCH_SIZE=1
EPOCHS=125
EVAL_NUM=400
RANDOM_FLAG=0
#TODO Watchout for Dropout ratio

ngc batch run \
--team "dlmed" \
--name "Unet_${EXP_NAME} ml-model.3DUnet" \
--preempt RUNONCE \
--image $NGC_CONTAINER \
--ace nv-us-west-2 \
--instance $NGC_INSTANCE \
--result /results \
--datasetid ${NGC_DATASET}:/dataset \
--workspace ${NGC_WORKSPACE} \
--commandline "export PYTHONPATH=${EXPORT_PATH};cd ${FOLDER_PATH};pip install -r requirements.txt --ignore-installed;python main.py --base_dir=${BASE_DIR} --data_root=/dataset/ --json_path=${JSON_PATH} --active_iters=5 --dropout_ratio=0.2 --mc_number=10 --queries=5 --random_strategy=${RANDOM_FLAG} --batch_size=${BATCH_SIZE}  --epochs=${EPOCHS}"
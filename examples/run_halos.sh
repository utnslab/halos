#!/bin/bash
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

source $SCRIPT_DIR/pythia_70m_commons.sh

RESULT_DIR=$HALOS_DATA_DIR/halos_results

# HALoS hyperparms
GLR=0.15
GBETA=0.5
GMUD=2
LLR=0.2
LBETA=0.9
LMUD=16
ALPHA=0.25
K=32
N_LOCAL_STEPS=8

_GLR=$(multiply $GLR $GMUD)
_LLR=$(multiply $LLR $LMUD)

mkdir -p $RESULT_DIR
python -m halos.simulation.exec_halos \
  --exp_name halos/halos_results \
  --result_dir $RESULT_DIR \
  --dataset_path $HALOS_DATA_DIR/datasets/pile-deduped-pythia-preshuffled/document \
  --env $SCRIPT_DIR/env.yaml \
  --model_name $MODEL_NAME \
  --global_batch_size $GLOBAL_BATCH_SIZE \
  --total_steps $TOTAL_STEPS \
  --micro_batch_size $MICRO_BATCH_SIZE \
  --num_gradient_accumulation $NUM_GRADIENT_ACCUMULATION \
  --local_step_time_ms $LOCAL_STEP_TIME_MS \
  --gps_opt_config "{\"opt_type\": \"delayed_nesterov\", \"lr\": $_GLR, \"beta\": $GBETA, \"buffer_size\": $GMUD }" \
  --num_lps 4 \
  --num_workers_per_lps 4 \
  --lps_opt_config "{\"opt_type\": \"delayed_nesterov\", \"lr\": $_LLR, \"beta\": 0.9, \"buffer_size\": $LMUD }" \
  --model_merge_weight $ALPHA \
  --local_updates_accumulation $K \
  --worker_lr_config "{\"lr_type\": \"cosine_decay_after_linear_warmup\", \"max_lr\": $WORKER_LR,  \"t_warmup\": $WARMUP_ITERS, \"t_max\": $TOTAL_ITERS }" \
  --worker_opt_config "{\"opt_type\": \"adamw\", \"lr\": $WORKER_LR, \"eps\": 1.0e-8, \"betas\": [0.9, 0.95], \"weight_decay\": 0.1 }" \
  --rescale_num_local_steps \
  --num_local_steps $N_LOCAL_STEPS \
  --val_interval_steps $VAL_INTERVAL_STEPS \
  --checkpointing_interval_steps $CHECKPOINTING_INTERVAL_STEPS \
  2>&1 | tee $RESULT_DIR/output.txt

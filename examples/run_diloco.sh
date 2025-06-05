#!/bin/bash
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

source $SCRIPT_DIR/pythia_70m_commons.sh

RESULT_DIR=$HALOS_DATA_DIR/diloco_results

# DiLoCo hyperparms
OUTER_LR=0.7
OUTER_BETA=0.9
N_LOCAL_STEPS=32

mkdir -p $RESULT_DIR
python -m halos.simulation.exec_diloco \
  --exp_name halos/diloco_results \
  --result_dir $RESULT_DIR \
  --dataset_path $HALOS_DATA_DIR/datasets/pile-deduped-pythia-preshuffled/document \
  --env $SCRIPT_DIR/env.yaml \
  --model_name $MODEL_NAME \
  --global_batch_size $GLOBAL_BATCH_SIZE \
  --total_steps $TOTAL_STEPS \
  --micro_batch_size $MICRO_BATCH_SIZE \
  --num_gradient_accumulation $NUM_GRADIENT_ACCUMULATION \
  --local_step_time_ms $LOCAL_STEP_TIME_MS \
  --outer_opt_config "{\"opt_type\": \"sgd\", \"lr\": $OUTER_LR, \"momentum\": $OUTER_BETA, \"nesterov\": true }" \
  --worker_lr_config "{\"lr_type\": \"cosine_decay_after_linear_warmup\", \"max_lr\": $WORKER_LR,  \"t_warmup\": $WARMUP_ITERS, \"t_max\": $TOTAL_ITERS }" \
  --worker_opt_config "{\"opt_type\": \"adamw\", \"lr\": $WORKER_LR, \"eps\": 1.0e-8, \"betas\": [0.9, 0.95], \"weight_decay\": 0.1 }" \
  --num_local_steps $N_LOCAL_STEPS \
  --val_interval_steps $VAL_INTERVAL_STEPS \
  --checkpointing_interval_steps $CHECKPOINTING_INTERVAL_STEPS \
  2>&1 | tee $RESULT_DIR/output.txt

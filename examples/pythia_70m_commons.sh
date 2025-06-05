multiply() {
    local num1=$1
    local num2=$2
    # Perform multiplication and format the result
    local result=$(echo "$num1 * $num2" | bc)
    printf "%.4f" "$result"
}

MODEL_NAME=pythia-70m-deduped
WORKER_LR=0.001
GLOBAL_BATCH_SIZE=1024
TOTAL_ITERS=6144
WARMUP_ITERS=300
TOTAL_BATCH_SIZE=$(($GLOBAL_BATCH_SIZE * $TOTAL_ITERS))

LOCAL_BATCH_SIZE=64
LOCAL_STEP_TIME_MS=238.391

MICRO_BATCH_SIZE=32
NUM_GRADIENT_ACCUMULATION=$(($LOCAL_BATCH_SIZE / $MICRO_BATCH_SIZE))

TOTAL_STEPS=$(($TOTAL_BATCH_SIZE / $LOCAL_BATCH_SIZE))
VAL_INTERVAL_STEPS=$((128 * $GLOBAL_BATCH_SIZE / $LOCAL_BATCH_SIZE))

# Set this to checkpoint intermediate models
CHECKPOINTING_INTERVAL_STEPS=-1

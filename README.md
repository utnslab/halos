# HALoS: Hierarchical Asynchronous Local SGD over Slow Networks for Geo-Distributed Large Language Model Training

This repository contains the geo-distributed LLM-training simulator used in our ICML 2025 paper ([HALoS](https://icml.cc/virtual/2025/poster/45594)).

## Installation

```bash
# 1. Create and activate a conda environment
conda create -n halos python=3.10.14
conda activate halos

# 2. Install dependencies and the HALoS package
pip install -r requirements.txt
pip install -r requirements-flashattn.txt
pip install -e .
```

## Prepare the deduped pile dataset (≈ 1.2 TB)
1. Choose a directory and export its path:
```bash
export HALOS_DATA_DIR=/path/to/data
```

2. Download [EleutherAI/pile-deduped-pythia-preshuffled](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled) into `$HALOS_DATA_DIR/datasets/pile-deduped-pythia-preshuffled`.

```bash
mkdir $HALOS_DATA_DIR/datasets && cd $HALOS_DATA_DIR/datasets
git lfs clone https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled
```

3. Merge the 20 shards into a single mem-map file using `utils/unshard_memmap.py` in [pythia](https://github.com/EleutherAI/pythia?tab=readme-ov-file#reproducing-training) repo:
```bash
python pythia/utils/unshard_memmap.py \
    --input_file $HALOS_DATA_DIR/datasets/pile-deduped-pythia-preshuffled/document-00000-of-00020.bin \
    --num_shards 20 \
    --output_dir $HALOS_DATA_DIR/datasets/pile-deduped-pythia-preshuffled
```

## Launch a local Ray cluster

``` bash
# Start the head node
ray start --head --port=6379

# (Optional) Add a worker node from another machine
ray start --address="$HEAD_ADDRESS"

# Shut down the cluster later (when you're done)
ray stop
```

## Run the examples
Each script trains Pythia-70M for 12.9B tokens and writes results to
`$HALOS_DATA_DIR/${ALGO_NAME}_results`.
If `$WANDB_API_KEY` is set, training and validation losses are logged to [Weights & Biases](https://wandb.ai/).

```bash
# HALoS (our method)
bash examples/run_halos.sh

# DiLoCo baseline
bash examples/run_diloco.sh

# DiLoCo + DynUpd baseline
bash examples/run_diloco_dynupd.sh

# Async-Local-SGD baseline
bash examples/run_async_local_sgd.sh
```

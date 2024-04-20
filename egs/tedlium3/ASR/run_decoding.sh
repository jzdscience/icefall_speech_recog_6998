#!/bin/bash

# run experiments
world_size=`nvidia-smi  -L | wc -l`

## baseline: all conv

## baseline: no conv/tdnn at all

## Variation on TDNN structure

### TDNN1 (512 3k 0d -  512 3k 2d - 512 3k 4d)

### TDNN2 (512 3k 0d -  512 3k 0d - 512 3k 0d)
 ./conformer_ctc2/decode.py --epoch 30  --exp-dir ./conformer_ctc2/exp_all_tdnn_tdnn2 --result-dir ./conformer_ctc2/exp_all_tdnn_tdnn2  --max-duration 400 --method ctc-decoding --use-averaged-model True --avg 5 --num-encoder-layers 4 --num-decoder-layers 2

### TDNN3 (512 5k 0d -  512 5k 2d - 512 5k 4d)

### TDNN4 (512 3k 0d -  512 3k 2d - 512 3k 4d - 512 3k 4d)

### TDNN5 (128 5k 0d -  128 5k 2d - 128 5k 4d)

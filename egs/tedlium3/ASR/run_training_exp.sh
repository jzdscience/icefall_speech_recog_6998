#!/bin/bash

# run experiments
world_size=`nvidia-smi  -L | wc -l`

## baseline: all conv
#./conformer_ctc2/train.py --world-size 4 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_test --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_conv

## baseline: no conv/tdnn at all
#./conformer_ctc2/train.py --world-size 4 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_no_conv --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type no_conv

## Variation on TDNN structure
### TDNN1 (512 3k 0d -  512 3k 2d - 512 3k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn1 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn1

### TDNN2 (512 3k 0d -  512 3k 0d - 512 3k 0d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn2 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn2

### TDNN3 (512 5k 0d -  512 5k 2d - 512 5k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn3 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn3

### TDNN4 (512 3k 0d -  512 3k 2d - 512 3k 4d - 512 3k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn4 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn4

### TDNN5 (128 5k 0d -  128 5k 2d - 128 5k 4d)
./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn5 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn5

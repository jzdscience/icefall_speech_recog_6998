#!/bin/bash

# run experiments
world_size=`nvidia-smi  -L | wc -l`

# ## baseline: all conv
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_conv_gcp --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_conv

# ## baseline: no conv/tdnn at all
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_no_conv --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type no_conv

# ## Variation on TDNN structure
# ### TDNN1 (512 3k 0d -  512 3k 2d - 512 3k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn1 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn1

# ### TDNN2 (512 3k 0d -  512 3k 0d - 512 3k 0d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn2 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn2

# ### TDNN3 (512 5k 0d -  512 5k 2d - 512 5k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn3 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn3

# ### TDNN4 (512 3k 0d -  512 3k 2d - 512 3k 4d - 512 3k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 18 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn4 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn4

# ### TDNN5 (128 5k 0d -  128 5k 2d - 128 5k 4d)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn5 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn5

# ### TDNN6 (512 3k 0d -  512 3k 0d - 512 3k 0d - Double Swish)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn6 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn6

# ### TDNN7 (512 3k 0d -  512 3k 0d - 512 3k 0d - Statspooling - Double Swish)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn7 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn7

# ### TDNN2NoSKIP (512 3k 0d -  512 3k 0d - 512 3k 0d )
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn2_noskip --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn_no_skip --tdnn-type tdnn2

# ### TDNN8 (176 3k 0d -  176 3k 0d - 176  3k 0d - Double Swish)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn8 --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_tdnn --tdnn-type tdnn8

# ### Conv_TDNN8 (c-c-t-t)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_tdnn8_conv_c_c_t_t --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type c_c_t_t --tdnn-type tdnn8

# ### Conv_TDNN8 (t-t-c-c)
# ./conformer_ctc2/train.py --world-size $world_size  --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_tdnn8_conv_t_t_c_c --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type t_t_c_c --tdnn-type tdnn8

# ### Conv_TDNN8 (c-t-t-c)
# ./conformer_ctc2/train.py --world-size $world_size  --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exps/exp_tdnn8_conv_c_t_t_c --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type c_t_t_c --tdnn-type tdnn8

### TDNN8 Full Model (176 3k 0d -  176 3k 0d - 176  3k 0d - Double Swish)
# ./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-epoch 5 --use-fp16 False --exp-dir conformer_ctc2/exps/exp_all_tdnn_tdnn8_full --max-duration 200 --num-encoder-layers 24 --num-decoder-layers 6 --conv-type all_tdnn --tdnn-type tdnn8 --save-every-n 4000

### Baseline Full Model
./conformer_ctc2/train.py --world-size $world_size --num-epochs 30 --start-batch 95000 --use-fp16 False --exp-dir conformer_ctc2/exps/exp_all_conv_full --max-duration 150 --num-encoder-layers 24 --num-decoder-layers 6 --conv-type all_conv --save-every-n 5000

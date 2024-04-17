# run experiments
#./conformer_ctc2/train.py --world-size 4 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exp_test --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type all_conv

./conformer_ctc2/train.py --world-size 4 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir conformer_ctc2/exp_no_conv --max-duration 400 --num-encoder-layers 4 --num-decoder-layers 2 --conv-type no_conv
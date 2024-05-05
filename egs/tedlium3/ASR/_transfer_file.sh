#!/bin/bash

# Local source directories
src1="conformer_ctc2/exps/exp_all_tdnn_tdnn2/"
src2="conformer_ctc2/exps/exp_all_conv/"

# Remote destination directory
dest1="e6998:~/icefall_6998_speech_recognition/egs/tedlium3/ASR/conformer_ctc2/exps/exp_all_tdnn_tdnn2/"
dest2="e6998:~/icefall_6998_speech_recognition/egs/tedlium3/ASR/conformer_ctc2/exps/exp_all_conv/"

# Rsync command with exclusion pattern
rsync -avz --include='epoch-25.pt' --include='epoch-26.pt' --include='epoch-27.pt' --include='epoch-28.pt' --include='epoch-29.pt' --include='epoch-30.pt' --exclude='*.pt' "$src1" "$dest1"
rsync -avz --include='epoch-25.pt' --include='epoch-26.pt' --include='epoch-27.pt' --include='epoch-28.pt' --include='epoch-29.pt' --include='epoch-30.pt' --exclude='*.pt' "$src2" "$dest2"
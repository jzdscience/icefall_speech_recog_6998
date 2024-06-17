#!/bin/bash

# Local source directories
src1="conformer_ctc2/exps/exp_all_conv_gcp/"

# Remote destination directory
dest1="e6998:~/icefall_6998_speech_recognition/egs/tedlium3/ASR/conformer_ctc2/exps/exp_all_conv_gcp/"

# Rsync command with exclusion pattern
rsync -avz --include='epoch-25.pt' --include='epoch-26.pt' --include='epoch-27.pt' --include='epoch-28.pt' --include='epoch-29.pt' --include='epoch-30.pt' --exclude='*.pt' "$dest1" "$src1"

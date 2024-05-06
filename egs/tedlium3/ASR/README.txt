Author: Ju Zhang, jz3702

Date: 05/04/2024
===================================================================================================================

Project Title: Using TDNN As An Alternative Convolution Strategy In Conformer-CTC E2E ASR Systems: A Performance And Efficiency Analysis

Project summary:

A significant advancement in end-to-end (E2E) ASR systems is the adoption of the Conformer-based encoder, renowned for its performance improvements. The Conformer encoder typically comprises several blocks, each featuring a "Macaron" structure with two feed-forward modules surrounding a multi-head self-attention module and a convolution module. The convolution module is crucial for capturing local features and enhancing the context understanding, contributing significantly to the Conformer's superior performance. This study explores the effects of replacing the convolution module with a Time Delay Neural Network (TDNN) module on training time and accuracy. Results indicate that a well-architected TDNN module, even when keeping the same number of parameters, not only enhances performance but also reduces training time. Additionally, this research examines the impact of mixed use of convolution-based Conformer blocks and TDNN-based Conformer blocks.

===================================================================================================================
List of tools:

Python, Anaconda, pytoch, CUCA, CuDNN, torchaudio, k2, icefall, lhotse

See ENVIRONMENT SET UP section for more details, including version information

===================================================================================================================
The study is based on the Conformer-CTC2 recipe of Tedlium3 dataset of the Icefall module in next-gen Kaldi project. 
https://github.com/k2-fsa/icefall

The repo was forked on March 16, 2024. Experiments are done via directly modifying and adding into the existing codebase of `conformer_ctc2` recipe.

==================================================================================================================
The file structure is illustrated as below,  asteriks *** shows where code addition/revision occurs

icefall_speech_recog_6998/
-egs/
--tedlium3/
---ASR/
----prepare.sh                         # call this to prepare data
----run_decoding.sh                    # *** new script *** shell script to run decoding process for all models, i.e. call decode.py in a loop
----run_training.sh                    # *** new script *** shell script to run training process for all models, i.e. call train.py in a loop
----download/
----data/
----conformer_ctc2/
--------README.txt                     # *** new script *** this file
--------conformer_ctc2.jz3702.diff     # *** new script *** a diff file showing all changed script in the conformer_ctc2/ folder
--------asr_datamodule.py              # *** revised***  a module to load data. Slightly modified as the cut_set.compute_and_store_features seems to have a bug that prevent the script running; switch to cut_set.compute_and_store_features_batch 
--------attention.py                   # define attention architecture
--------conformer.py                   # *** revised***  confomer model architecture. This is the major script being modified. Adding new classes of TDNN modules, and new class of Conformer blocks, etc
--------decode.py                      # *** revised***  script to call for decoding. Modified as some additional arguments need to pass into it.
--------exps/                          # experiment results, including checkpoint, log and tensorboard files are saved here; each subfold is for one experiment.
--------export.py                      # functionality to export model
--------label_smoothing.py             # functionality to do label_smoothing
--------lstmp.py                       # LSTM with projection, not used 
--------optim.py                       # optimzer definition
--------scaling_converter.py           # This file provides functions to convert scaled layers to non-scaled counterparts
--------scaling.py                     # Here defines classes of activation functions, scaled convolution, scaled embedding etc
--------subsampling.py                 # 2D convolutional subsampling model architecture
--------train.py                       # *** revised*** script to call for model training. Modified as some additional arguments need to pass into it.
--------transformer.py                 # transformer model architecture

for detailed code difference from original egs/tedlim3/ASR/conformer_ctc2, please check conformer_ctc2.jz3702.diff 

=====================================================================================================

To run the code, you need to step up the Kaldi envrionment first.
WARNING: It is extremely fragile. May take a few hours even days to set up and run expriment without any issue. 
I am very happy to help to trouble shooting!

I. ENVIRONMENT SET UP

To have the Same envrionment I have:

1. Spin a Google Cloud VM 
1.1 Choose N1- 8vCPU 52GB RAM, 1200GB HDD with 4 T4 GPU
1.2 Choose Ubuntu 20.04 LTS as OS, not pre-built deep-learning environment 

2. Launch

3. SSH to the VM and let us start install things:
3.1. Install NVIDIA driver https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installdriver
After this, the `nvidia-smi` command should work...

3.2.  Install CUDA toolkit and CuDNN
3.2.1 Need to apt-get install build-essentials (make) and gcc to install cuda
3.2.2 Install CUDA toolkit 12.1 and CuDNN following https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
After this, the `nvcc --version` command should work
WARNING: Do not use pip/conda install cudatoolkit

3.2.3 Make a activate activate-cuda-12.1.sh  https://k2-fsa.github.io/k2/installation/cuda-cudnn.html#cuda-11-6
3.2.4 bash ~/activate-cuda-12.1.sh 

3.3. Install Anaconda to manage the virtual environment
3.3.1 Create an virtual enviroment with 
conda create -n speech_recog python=3.11.7
3.3.2 Activate the virtual envrionment
conda activate speech_recog

3.4. Install ffmpeg. conda install -c conda-forge 'ffmpeg<7'

3.5 Install the rest of packages following the *Installation Example* on https://icefall.readthedocs.io/en/latest/installation/index.html#installation-example
which includes: 
3.5.1 Torch (2.2.1 for CUDA 12.1) and torchaudio (2.2.1 for CUDA 12.1) at the same time from https://download.pytorch.org/whl/torch_stable.html
    pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
3.5.2 K2 (1.24.4 for CUDA 12.1 and Torch 2.2.1) from https://k2-fsa.github.io/k2/cuda.html
    wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20240301+cuda12.1.torch2.2.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install k2-1.24.4.dev20240301+cuda12.1.torch2.2.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
3.5.3 Lhotse
    pip install lhotse=1.21
pip install git+https://github.com/lhotse-speech/lhotse


3.6.1 enter my folder uploaded as code submission
(alternatively) download Icefall and install the requirements in the repo
git clone git@github.com:jzdscience/icefall_speech_recog_6998.git   # this is my fork from https://github.com/k2-fsa/icefall on March 16, 2024

3.6.2 Install Dependency
pip install -r ./icefall_6998_speech_recognition/requirements.txt


II. RUN THE EXPERIMENT

A. Change some envrionment variables
1. put my project path into PYTHONPATH
export PYTHONPATH=/home/ipsoct4/E6998/icefall_6998_speech_recognition/:$PYTHONPATH
2. 
2.1 (option 1) Hide CUDA device so decoding will run on CPU
export CUDA_VISIBLE_DEVICES=""
2.2 (option 2) Show CUDA device so training/decoding will run on GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3"

B. Prepare data
1. cd to `icefall_speech_recog_6998\egs\tedlium3\ASR\
2. call `./prepare.sh` to prepare data
basically, it will first download voice data, lexicon, lm, etcs into `icefall_speech_recog_6998\egs\tedlium3\ASR\download\` ... then prepare and save data to
`icefall_speech_recog_6998\egs\tedlium3\ASR\data\`

WARNING1: You do not need to do anything else unless it throw error, which is very likely!
WARNING2: Even everything is correct, data preparation takes 5-8 hours

C. Training (Optional)
1. cd to icefall_speech_recog_6998\egs\tedlium3\ASR\

2. call './run_training.sh', it should start to train for all models. 
WARNING: 
2.1 It take ~37min*30 epoch* 14 models = 260hrs to finish all training on a VM with 4 Tesla T4 GPU
2.2 You can reduce the max-duration is your vRAM is lower than 16GB, otherwise it is not enough.
2.3 (optional) You can skip training by downloading my model as in section D -1.1

D. Tracking (Optional)
Tracking is done by calling `tensorboard --log_dir=exps/`. The training time information was directly from Tensorboard

E. Decoding 
1. If you skipped traininig, make sure all subfolders in icefall_speech_recog_6998\egs\tedlium3\ASR\conformer_ctc2\exps\ have required checkpoints of epoch25-30 (as we use last 5 epoch's average model)

2. Call `run_decoding.sh` to decode using all models
NOTE: It takes ~8 mins for one model's decoding on CPU and 30 seconds on GPU; and we have 14 models.

3. The results of WER will be saved as `wer.summary.clean.txt` `wer.summary.other.txt` in individual run folder, for example,  `/exps/exp_all_tdnn_tdnn8/`

Hopefully by now you would have the results shown in my paper, except you need to manually calculate the training Time per Epoch (TE) by putting the number from Tensorboard in a spreadsheet or so....

===========================================================================================================================
Thank you so much! Please sent me an email jz3702@columbia.edu if you encounter any problem!
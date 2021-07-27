#!/bin/sh

# this will submit all the current submission scripts

cp slurm_submission_scripts/* .

sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_submissions.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_-1_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_-1_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_-1_submissions.sub

sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_v2_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_v2_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_v2_submissions.sub

sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_v3_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_v3_submissions.sub
sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_v3_submissions.sub

rm *_submissions.sub

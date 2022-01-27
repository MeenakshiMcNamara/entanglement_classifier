#!/bin/sh

# this will submit all the current submission scripts

cp slurm_submission_scripts/* .

sbatch -F hammer --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzdelta_gen_submissions_weighted_drop_0.2_no_qg.sub
sbatch -F hammer gpu --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzdelta_gen_submissions_weighted_drop_0.5_no_qg.sub

rm *_submissions.sub
rm *_submissions_no_neg.sub
rm *weighted*.sub
rm *qg.sub

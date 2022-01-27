#!/bin/sh

# this will submit all the current submission scripts

cp slurm_submission_scripts/* .

sbatch -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzdelta_gen_submissions_weighted_drop_0.2_no_qg.sub
sbatch -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzdelta_gen_submissions_weighted_drop_0.5_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentz_mttbar_submissions_weighted_drop_0.2_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentz_mttbar_submissions_weighted_drop_0.2_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentz_mttbar_submissions_weighted_drop_0.2_no_qg.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentz_mttbar_submissions_weighted_drop_0.5_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentz_mttbar_submissions_weighted_drop_0.5_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentz_mttbar_submissions_weighted_drop_0.5_no_qg.sub


#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentz_submissions_weighted_drop_0.5_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentz_submissions_weighted_drop_0.5_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentz_submissions_weighted_drop_0.5_no_qg.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentz_submissions_weighted_drop_0.2_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentz_submissions_weighted_drop_0.2_no_qg.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentz_submissions_weighted_drop_0.2_no_qg.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentzgen_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentzgen_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzgen_submissions_weighted_drop_0.5.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentzgen_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentzgen_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentzgen_submissions_weighted_drop_0.2.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_spinCorr_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_spinCorr_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_spinCorr_submissions_weighted_drop_0.5.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_lorentz_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_lorentz_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_lorentz_submissions_weighted_drop_0.2.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_submissions.sub

# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions.sub
# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions.sub
# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions.sub

# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions_no_neg.sub
# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions_no_neg.sub
# sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions_no_neg.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions_weighted.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions_weighted.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions_weighted.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions_weighted_drop_0.2.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions_weighted_drop_0.2.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions_weighted_drop_0.35.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions_weighted_drop_0.35.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions_weighted_drop_0.35.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.9_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.9_submissions_weighted_drop_0.5.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.9_submissions_weighted_drop_0.5.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_-1_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_-1_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_-1_submissions.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_v2_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_v2_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_v2_submissions.sub

#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 mumu_cut_0.8_v3_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 emu_cut_0.8_v3_submissions.sub
#sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 ee_cut_0.8_v3_submissions.sub

rm *_submissions.sub
rm *_submissions_no_neg.sub
rm *weighted*.sub
rm *qg.sub

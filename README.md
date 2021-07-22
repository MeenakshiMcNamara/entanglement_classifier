# entanglement_classifier
Create a NN classifier for determining ttbar production mode

The saved models after runs are in /models 

The code which can be used to analyze the results and look for correlations is in /analysis_code

slurm submission scripts (used for submitting jobs to run on the clusters) are in /slurm_submission_scripts. Use variant of `sbatch  -A long --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 submissions.sub` command to run them (note that 4 hours is the max for scholar gpu sessions)

code_to_import contains python files which are only used when imported in a variety of places. Note that to use these in a sister folder you should use `import sys` and `sys.path.append(<path to entanglement_classifier directory")` and then append `code_to_import` to file name in import

slurm_output should contain all the outputs that you probably don't care about anymore

/data contains the numpy array data for the training loss and validation loss

/<insert number>_layer_classifiers contain the code which creates and runs the classifiers
  
  /data_adders contains the code for generating the root files and data frames with the event data being trained on

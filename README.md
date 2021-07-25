# entanglement_classifier
The main purpose of this repo is to create a neural network classifier for determining ttbar production mode between gg, qqbar, and other. Below is a short description of the organizational scheme of the repo:


*****
The saved models after runs are in /models note that they are saved separately by the number of layers in the netowork, and that I created a folder in /two_layers for older versions that are no longer relevent. In the future, optimizer states will also be saved in /models.

The code which can be used to analyze the results and look for correlations is in /analysis_code. This includes confusion matrices, correlation matrices, and both the derivative and permutation methods of determining importance, as well as any other code used to analyze the data from runs. Results from analysis go into the /results subfolder. Try to keep these as organized as possible by making any relavent subfolders in it to group together information. Note that I have put copies of the files (including ones for running stuff in parallel) in a separate (less organized) subfolder.

slurm submission scripts (used for submitting jobs to run on the clusters) are in /slurm_submission_scripts. Use a variant of the `sbatch  -A scholar --nodes=1 --gres=gpu:1 --gpus=1 -t 04:00:00 submissions.sub` command to run them (note that 4 hours is the max for scholar gpu sessions). To view your slurm jobs use `squeue -u <username>` and to cancel jobs use `scancel <jobid>`.

/code_to_import contains python files which are only used when imported in a variety of places. At the moment these are the dataset_preprocessing and Classifier_module. Note that to use these in a sister folder you should use `import sys` and `sys.path.append("<path to entanglement_classifier directory>")` and then prepend `code_to_import` to file name in import.

/slurm_output should contain all the outputs that you probably don't care about anymore, but want to keep just in case.

/data contains the numpy array data for the training loss and validation loss.

/[insert number]_layer_classifiers contain the code which creates and runs the classifiers. Put older versions you want to save in subdirectories to keep things organized.

/data_adders contains the code for generating the root files and data frames with the event data being trained on.

### How to run analysis (examples using 2016 pre-ultralegacy)
In order to do this, you must be in the `/dileptonic` directory in one of the clusters which give you cms access.
1.	Run  `cmsenv`. I recommend adding both the path to the most used directory (`/dileptonic` for me) and this command to your .bash_profile/ whatever you have.
2.	From inside `/dileptonic`, run `./scripts/compileAll.sh` if you have made any changes to the C++ code which you want to include in the run.
3.	**First time on ntuple only:** Run the btags scripts.
    a.  ***Warning:*** The btag scripts will run using the same commands as the normal analysis commands if the file in '/selectionRoot_2016' doesn’t exist. Make sure you use th entire dataset if running btags.
    b.	Recommended commands are 
    `nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c ee -s Nominal &> nohuplogs/BTag_ee.out &
     nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c emu -s Nominal &> nohuplogs/BTag_emu.out &
     nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c mumu -s Nominal &> nohuplogs/BTag_mumu.out &`
     The `nohup` argument causes the `./install/bin/load_Analysis` script to run in the background, `-f` specifies the ntuple to use, `-c` specifies the channel as ee, emu or mumu (this document does not explain how to implement tau), and `nohuplogs/BTag_[channel].out` is the output file for the log. You might consider putting these commands in a bash script (usually a file ending in .sh) using `#!/bin/bash` at the beginning of the file to speed the process of running these commands.
    c.   Run `cp -r selectionRoot_2016/BtagEff_2016 .` after it has finished
4.	Run the analysis scripts

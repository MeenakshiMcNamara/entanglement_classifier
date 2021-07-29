How to run analysis (examples using 2016 pre-ultralegacy)
1.	Run  `cmsenv`. I recommend adding both the path to the most used directory (`/dileptonic` for me) and this command to your .bash_profile/ whatever you have.
2.	From inside `/dileptonic`, run `./scripts/compileAll.sh` if you have made any changes to the C++ code which you want to include in the run.
3.	**First time on ntuple only:** Run the btags scripts.
    a.  ***Warning:*** The btag scripts will run using the same commands as the normal analysis commands if the file in '/selectionRoot_2016' doesn’t exist. Make sure you use th entire dataset if running btags.
    b.	Recommended commands are 
    `nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c ee -s Nominal &> nohuplogs/BTag_ee.out &
     nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c emu -s Nominal &> nohuplogs/BTag_emu.out &
     nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c mumu -s Nominal &> nohuplogs/BTag_mumu.out &`
     You might consider putting them in a bash script (usually a file ending in .sh) using `#!/bin/bash` at the beginning of the file to speed the process
    c.   Run `cp -r selectionRoot_2016/BtagEff_2016 .` after it has finished
4.	Run the analysis scripts
5.	

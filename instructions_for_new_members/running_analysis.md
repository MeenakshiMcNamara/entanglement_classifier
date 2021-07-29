# How to run analysis (examples using 2016 pre-ultralegacy without signalviatau)
In order to do this, you must be in the `/dileptonic` directory in one of the clusters which give you cms access.
1.	Run  `cmsenv`. I recommend adding both the path to the most used directory (`/dileptonic` for me) and this command to your .bash_profile/ whatever you have.
2.	From inside `/dileptonic`, run `./scripts/compileAll.sh` if you have made any changes to the C++ code which you want to include in the run.
3.	**First time on ntuple only:** Run the btags scripts.
    1.  ***Warning:*** The btag scripts will run using the same commands as the normal analysis commands if the file in '/selectionRoot_2016' doesn’t exist. Make sure you use th entire dataset if running btags.
    1.	Recommended commands are:  
      
        `nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c ee -s Nominal &> nohuplogs/BTag_ee.out   
        nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c emu -s Nominal &> nohuplogs/BTag_emu.out &  
        nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -c mumu -s Nominal &> nohuplogs/BTag_mumu.out &`  
          
        The `nohup` argument causes the `./install/bin/load_Analysis` script to run in the background, `-f` specifies the ntuple to use, `-c` specifies the channel as ee, emu or mumu (this document does not explain how to implement tau), and `nohuplogs/BTag_[channel].out` is the output file for the log. You might consider putting these commands in a bash script (usually a file ending in .sh) using `#!/bin/bash` at the beginning of the file to speed the process of running these commands.
    2.   Run `cp -r selectionRoot_2016/BtagEff_2016 .` after it has finished (you can check if a background process has finished by running `top` which shows the running programs using the most resources and `ps aux | grap [username]` to see all processes you have running. Check for anything with `./install/bin/load_Analysis`.)
4.	Run the analysis scripts. (Please read the above warning). Simarly to the Btag scripts, run the following (which I recommend putting in a bash script).  
  
    `nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -m spinCorr -c ee &> nohuplogs/SpinCorr_ttbarsignalplustau_ee.out &  
    #nohup ./install/bin/load_Analysis -f ttbarsignalplustau.root -m spinCorr -c ee --signalviatau &> nohuplogs/SpinCorr_ttbarsignalviatau_ee.out &  
    nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -m spinCorr -c emu &> nohuplogs/SpinCorr_ttbarsignalplustau_emu.out &  
    #nohup ./install/bin/load_Analysis -f ttbarsignalplustau.root -m spinCorr -c emu --signalviatau &> nohuplogs/SpinCorr_ttbarsignalviatau_emu.out &  
    nohup ./install/bin/load_Analysis -f ttbarsignalplustau_fromDilepton_TuneCP5.root -m spinCorr -c mumu &> nohuplogs/SpinCorr_ttbarsignalplustau_mumu.out &  
    #nohup ./install/bin/load_Analysis -f ttbarsignalplustau.root -m spinCorr -c mumu --signalviatau &> nohuplogs/SpinCorr_ttbarsignalviatau_mumu.out &`  
      
    The only difference is where the output is saved to ensure you can debug both the BTag and ananlysis. The lines which are commented out are for tau, which is beyond the scope of the document.
        1. To cancel jobs, run `kill -9 [job-id]`.
  
-------------------------------------------------------------------
  
# How to edit cuts in analysis run

This is actually very easy, but be careful because the original cuts were chosen for a reason. You should probably keep a copy of the original version just in case. Alright, warning given, the way to change the cuts is simply modify `config.txt` which is already inside of the `/dileptonic`. The cuts are towards the bottom of the file and are largely self explanatory. 
If there are any cuts you are unsure about the meaning of or you are curious about particle flow IDs (pfids) then a good resource is [pdg](https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf) (I have linked an example web page). pdg has good resources for many things in particle physics.

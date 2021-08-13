# Editing/ Adding Variables to Minitrees
There is quite a lot of variation that can be done for this depending on what you are trying to achieve, but these instructions contain the most common types of changes I (Meenakshi) have had to make, and should hopefully give you a better idea of how to do similar things as well. For instance, the variables will always need to be added to the minitrees, defined in all the correct places, and given values. How you calculate those values depends on what you are trying to do.

One other thing before you begin, please try to follow the general grouping of variables which already exists (top stuff is together...) to keep the code more easily usable.

**Note:** These instructions are specifically for the dileptonic, but much/ most of this will hopefully carry over to other parts of the framework as well.

1.  Declare the variables and variables names in the VariablesSpinCorr class declaration in `/include/Variables/SpincCorr.h`. (In C++, classes and methods are often used in a .h file which can then be "imported" using `#include <file.h>` to keep the code cleaner and allow it to run more quickly.) The variables are public so they must be declared after line 37 (probably not until line 65 at the earliest though), and the names are private and should be declared after line 363. Follow the same format as the declarations for existing minitree variables.  
2.  Create and import the variable branch in `/src/TreeHandlersSpinCorr.cc`. `VariablesSpinCorr.h` was included in this file, so we can use the class here. Put the branch creation in the `TreeHandlerSpinCorr::bookBranches` method (starting on line 49), and the branch import statement in the `TreeHandlerSpinCorr::importBranches` method (starting on line 367). Once again, follow the existing format for other minitree variables. 
3.  Set up method initialization for VariablesSpinCorr constructors in `src/VariablesSpinCorr.cc`. First do this somewhere after line 29 (follow the format shown) in the empty constructor initialization and then do this again inside of the regular constructer initialization which is started on line 326. Both of these should be done between the single : after the parameters are given to the constructor and before the body of it. Once again, follow the format shown.
4.  This is the step where we define the value of the variable you are adding. We only do this in the body of the non-empty constructor, which means after line 625. A number of TLorentzVectors are defined early on, as well a VariablesPhiTT object called vars which contains the spin correlation data (if you want to add variables from that some extra steps are involved), and most of the variables are filled after these are defined. The general syntax is `variable.value_ = [whatever you want]`. Often there are examples of how to get the value you are interested in present in the same group you are adding code to, but sometimes you have to do something different. Probably the most common variables are just some part of a lorentz vector or part of vars (if you want to add something to vars or something similar then you will need to determine where it is defined in a different file and add stuff to the corresponding .h file as needed... VS Code has some useful tools for finding where things are declared/defined), but there are many other things you can do including operations with lorentz vectors. I often prefer to replace TLorentzVectors with LV because the original one has been deprecated, and if you like to do this there are simple `LVtoTLV` and `TLVtoLV` methods which make this easy. I am linking the documentation for `LorentzVector` (aka LV) class [here](https://root.cern/doc/v616/LorentzVectorPage.html) for you.
5.  Recompile everything so your changes will show up when you rerun the analysis. The command to do this is `./scripts/compileAll.sh` (this takes a while, so you may want to do this using `nohup` to place in the background). If you are like me, you will probably run into a few compile errors along the way :wink:, but the output will let you know where they are, so it should help with trouble shooting.

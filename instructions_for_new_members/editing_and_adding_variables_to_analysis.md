# Editing/ Adding Variables to Minitrees
There is quite a lot of variation that can be done for this depending on what you are trying to achieve, but these instructions contain the most common types of changes I (Meenakshi) have had to make, and should hopefully give you a better idea of how to do similar things as well. For instance, the variables will always need to be added to the minitrees, defined in all the correct places, and given values. How you calculate those values depends on what you are trying to do.

One other thing before you begin, please try to follow the general grouping of variables which already exists (top stuff is together...) to keep the code more easily usable.

Note: These instructions are specifically for the dileptonic, but much/ most of this will hopefully carry over to other parts of the framework as well.

1.  Declare the variables and variables names in the VariablesSpinCorr class declaration in `/include/Variables/SpincCorr.h` The variables are public so must be declared after line 37 (probably not until line 65 at the earliest though), and the names are private and should be declared after line 363. Follow the same format as the declarations for existing minitree variables.  
2.  Create and import the branch in `/src/TreeHandlersSpinCorr.cc` (around lines 158 and 446). Once again, follow the existing formats.  
3.  Set up method initialization for VariablesSpinCorr constructor in `src/VariablesSpinCorr.cc`. First do this somewhere after line 29 (follow the format shown) in the empty constructor and then do this again inside of the constructer which will actually be used which is started on line 326. Once again, follow the format shown. Note that in both of these cases you should try to follow the grouping structure which already exists   
4.  This is the step where we define the value of the variable you are adding. Often there are examples of how to get the value you are interested in present in the same group you are adding code to, but sometimes you have to do something different. All of this code will be added to the method  

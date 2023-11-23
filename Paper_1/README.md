# Introduction
In this document we describe how to reproduce the results given in the paper entitled "Harnessing the Quantum Zeno Effect: A new Approach to Ion Trapping", authored by Varqa Abyaneh. There are 3 python files:

* quantum_functions.py
* main.py
* wavefunction_bound.py

# Setup
Here are some setup remarks


# Running
Here are some remarks on running the code

## quantum_functions.py
This is library of functions. The functions are called from main.py. There are more functions in the library than are called from main.py, as the library is being used for other research purposes as well. 

## main.py
This code is used to produce the results (on the tabs "Figure 4 & Table 2 information", "Figure 5 & Table 3 information", and "Convergence TestinG") in the accomanying spreadsheet called Results_and_Conververgence_Testing.xls. Unless stated on the spreadsheet, the "Setup" parameters are not to be changed.  Below is description of each tab

*Tab: Figure 4 & Table 2 information*
The input data is given on the top table. These can paramters can be put into the "Setup" parameters and then the code can be run. You will obtain the relevant information.

* Plot of the wavefunction using python graphics (Figure 4)
* Raw data for the wavefunction (Figure 4)
* Printed Energy eigenvalue (Table 2)

*Tab: Figure 5 & Table 3 information*
The input data is given on the top table. These can paramters can be put into the "Setup" parameters and then the code can be run. You will obtain the relevant information.

* Printed leakage (calibration impacting Table 3 results)
* Printed leakage (calibration impacting plot of Figure 5)

*Tab: Convergence Testing*
The input data is given in the tables. These can paramters can be put into the "Setup" parameters and then the code can be run. You will obtain the relevant information outputs in the table showing the convergence analysis. 
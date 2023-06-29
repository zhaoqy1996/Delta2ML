## $\Delta^2$ Machine Learning for Reaction Property Prediction

The emergence of $\Delta$-learning models, whereby machine learning (ML) is used to predict a correction to a low-level energy calculation, provides a versatile route to accelerate high-level energy evaluations at a given geometry. However, $\Delta$-learning models are inapplicable to reaction properties like heats of reaction and activation energies that require both a high-level geometry and energy evaluation. Here, the $\Delta^2$-model is introduced that can predict high-level activation energies based on low-level critical-point geometries. 

This project provides a pytorch implementation of a $\Delta^2$-learning model that uses GFN2-xTB level optimized geometries and corresponding single point energies to provide DFT (B3LYP-D3/TZVP) and Gaussian-4 (G4) level single point energies. This model is trained on both equilibrium structures (e.g., reactant and product) and transition states thus can be used to predict activation energies for C,H,O,N-containing systems.

### Minimum Anaconda Package Requirement 
* python>=3.7
* pandas=1.2.4 
* numpy=1.23.4
* pytorch>=1.10.0
  
You can simply install the anaconda environment by
```
conda env create -f environment.yml
```

### Basic Usage
1. Put xyz files of the your target geometries into a folder (e.g., input\_geo) which contains GFN2-xTB optimized geometries of either transition state or individual reactant and product (or both). For multi-molecular reactions, like A+B-->C, we recommend optimize A and B seperately. NOTE: All the input and output energies should be/are in Hartree!!!

2. Prepare a csv file containing 'name' (match with file name in the input geometry folder) and 'xTB\_ene' (xTB level single point energy). 

3. Run $\Delta^2$-ML prediction by (-l DFT or -l G4 are avaiable)

```
python predict.py -g input_geo -e xTB_energy.csv -l DFT -o output.csv
```
4. One specific example is:

```
python predict.py -g examples/YARPv2/input_geo/ -e examples/YARPv2/xTB_energy.csv -l DFT -o test_YARP2.csv
```

### Computing the activation energy
To compute the activation energy, users need to first obtain the GFN2-xTB optimized reactant(s) and transition state(s) and then apply the $\Delta^2$ model to obtain more accurate energies. In the end, simply substracting the (summation of) reactant energies by the transition state energy. Here provide one example that computes the activation energies of 3-(Hydroperoxy)propanal.

```
python predict.py -g examples/KHP_barriers/input_geo/ -e examples/KHP_barriers/xTB_energy.csv -l DFT -o KHP_test.csv
```

In this example, the reactant is named as 'XSASRUDTFFBDDK' and all other geometries are transition states. A simple python script to parse the activation energies is provided:

```
python parse_DE.py
```

The output looks like:

Activation energy of XSASRUDTFFBDDK_38_0_0 is  63.14 kcal/mol

Activation energy of XSASRUDTFFBDDK_24_0_1 is  82.20 kcal/mol

Activation energy of XSASRUDTFFBDDK_34_1_0 is  39.24 kcal/mol

...

### Computing the enthalpies of reaction

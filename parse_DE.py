import pandas as pd

df = pd.read_csv('KHP_test.csv')

# in this case, the reactant is named by XSASRUDTFFBDDK
RE = float(df[df.name=='XSASRUDTFFBDDK']['pred_SPE'].item())

# compute the activation energies
for i in range(len(df)):
    TS = df.iloc[i]
    if TS['name'] == 'XSASRUDTFFBDDK': continue
    print("Activation energy of {} is {:< 5.2f} kcal/mol".format(TS['name'], 627.5*(TS.pred_SPE-RE)))

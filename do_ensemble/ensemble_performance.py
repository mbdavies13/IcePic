'''
Script to test how a ensemble of models performs in regression task

Note: only the best ensmble has been uploaded to Apollo - so if using this to
recreate the data from the paper then the final largest ensemble should be the best one.

However if you train your own models then the below can be used to see what the best ensemble
is from the group of models you've trained.
'''

import pandas as pd
import glob
from general.code import *

### Directory to where ensemble of models is stored ###
STRAT_BOOL = True # whether or not applying to stratified (Fig.1) or specific split (Fig. 2)

# Args below are for apollo repository structure
if STRAT_BOOL:
    ## Stratified splits
    STRAT_N =  1 # 1, 2, 3, 4, 5, or 6
    DIR = '../../CNN_Models/ProductionModel/STRAT{}'.format(STRAT_N)
    STRAT_FILE = 'strat_index.{}.txt'.format(STRAT_N)

else:
    ## non stratified splits
    # options from paper fig 2:
    # SQ, HE, TR, OC, LJ_fcc100, LJ_fcc111, LJ_fcc110, LJ_fcc211, GR
    SPLIT_STR = 'SQ'
    DIR = '../../CNN_Models/TestingModels/{}'.format(SPLIT_STR)
    STRAT_FILE = None

# Ensure below is correct path to the model names
models = glob.glob('{0}/models/mod*'.format(DIR))
######################################################################

####### 1) GET TEST DATASET
IM_SIZE = 50 # size of images

## Read in dataframe
df = pd.read_csv('../dataframe.txt', sep=" ",header=0)
## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

#### Split into test/train
if STRAT_BOOL:
    f=open('{0}/{1}'.format(DIR, STRAT_FILE), "rb")
    train_index, test_index = pickle.load(f)
    f.close()
    df_train_stratified = df.iloc[train_index]
    df_test_stratified = df.iloc[test_index]

else:
    trainSYS = []
    testSYS = []
    for i, val in enumerate(df['sysname']):
        if SPLIT_STR in val:
            testSYS.append(i)
        else:
            trainSYS.append(i)
    df_train_stratified = df.iloc[trainSYS]
    df_test_stratified = df.iloc[testSYS]

# Make test dataset
X_test, Y_test = make_database_simple_w_resize(df_test_stratified,
                                               target='Tnuc_norm',
                                               new_size=(IM_SIZE,IM_SIZE),
                                               zpredFILE='wat_zpred.txt')

# reshape
X_test = X_test.reshape(X_test.shape[0], IM_SIZE, IM_SIZE, 1)

####### 2) GET MODELS AND SEE HOW ENSEMBLE PERFORMS
from keras.models import load_model

models_filterd_ld = []
models_filterd = np.sort(models)
# load models after sort - careful to keep ordering same
for m in models_filterd:
    models_filterd_ld.append(load_model(m))
print('\n###### Models from dir:\n', models_filterd)

## Evaluate total ensemble performance over individual performance
metrics = ensemble_performance(models_filterd_ld, X_test, Y_test,
                     preloaded=True, return_metrics=True)

# Now see how ensembles of different size perform
# note: only the best ensmble has been uploaded to Apollo - so if using this to
# recreate the data from the paper then the final largest ensemble should be the best one
# However if you train your own models then the below can be used to see what the best ensemble
# is from the group of models you've trained.
for n in np.arange(1,11,1):
    print(n)
    # Take the best n models based of off the RMSE score they achieved
    ensem_metrics = ensemble_performance(models_filterd[np.argsort(metrics[:,1])[:n]],
                         X_test, Y_test, preloaded=False, return_ensem_metrics=True)
    print('\n{0} {1}\nMAE: {2} RMSE: {3}\n{4}\n\n'.format(n,
                                                        ensem_metrics,
                                                        mae_from_maenorm_Tn(ensem_metrics[0], df),
                                                        rmse_from_msenorm_Tn(ensem_metrics[1], df),
                                                        models_filterd[np.argsort(metrics[:,1])[:n]])
          )


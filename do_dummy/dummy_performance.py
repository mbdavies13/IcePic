import numpy as np
import pandas as pd
import scipy
import pickle


## general code
from general.code import normalise_Tn, make_database_simple_w_resize

# Dummy models
from sklearn.dummy import DummyRegressor
from models.dummy import get_regression_metrics

####### 0) Decide which train/test going to access
## Find directory where ensemble of models is stored
STRAT_BOOL = True # whether or not applying to stratified (Fig.1) or specific split (Fig. 2)

# Args below are for apollo repository structure
if STRAT_BOOL:
    ## Stratified splits
    STRAT_N = 5 # 1, 2, 3, 4, 5, or 6
    #../CNN_Models/ProductionModel/
    STRAT_FILE = '../../CNN_Models/ProductionModel/STRAT{0}/strat_index.{0}.txt'.format(STRAT_N)

else:
    ## non stratified splits
    # options from paper fig 2:
    # SQ, HE, TR, OC, LJ_fcc100, LJ_fcc111, LJ_fcc110, LJ_fcc211, GR
    SPLIT_STR = 'SQ'


####### 1) Make TRAIN/TEST dataset
IM_SIZE = 50 # size of images
## Read in dataframe
df = pd.read_csv('../dataframe.txt', sep=" ",header=0)
## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

## Split into test/train
if STRAT_BOOL:
    f=open(STRAT_FILE, "rb")
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

## MAKE TRAIN AND TEST DATABASE
X_train, Y_train = make_database_simple_w_resize(df_train_stratified,target='Tnuc_norm',
                                                 zpredFILE='wat_zpred.txt',
                                                new_size=(IM_SIZE,IM_SIZE))

X_test, Y_test = make_database_simple_w_resize(df_test_stratified,target='Tnuc_norm',
                                                zpredFILE='wat_zpred.txt',
                                                new_size=(IM_SIZE,IM_SIZE))

## RESHAPE
print("Shape of training database prior to reshape:",X_train.shape)
X_train = X_train.reshape(X_train.shape[0], IM_SIZE, IM_SIZE, 1)
X_test = X_test.reshape(X_test.shape[0], IM_SIZE, IM_SIZE, 1)
print("Shape of training database after reshape:",X_train.shape)
print("Range of values of labels:", [np.min(Y_train), np.max(Y_train)])

##### DUMMY MODELS
# Build DummyRegressors
dummyregressor_mean = DummyRegressor(strategy='mean')
dummyregressor_median = DummyRegressor(strategy='median')
dummyregressor_mode = DummyRegressor('constant',
                                     scipy.stats.mode(np.array(df['Tnuc_norm']))[0][0])
# Fit Dummy Regressors
dummyregressor_mean.fit(X_train, Y_train)
dummyregressor_median.fit(X_train, Y_train)
dummyregressor_mode.fit(X_train, Y_train)


####
print('\n######\nFitting dummy models...')
dummy_regressors = [
    ('mean', dummyregressor_mean),
    ('median', dummyregressor_median),
    ('mode', dummyregressor_mode)
]
### calculate metrics (mae, mse, max error) for the two different models (=regressors) on both train and test sets.
dummy_regressor_results_test = {}  # initialize empty dictionary
for regressorname, regressor in dummy_regressors:
    dummy_regressor_results_test[regressorname] = get_regression_metrics(
        regressor, X_test, Y_test)

### rescaled metrics
dummy_regressor_results_test_rs = {}  # initialize empty dictionary
for regressorname, regressor in dummy_regressors:
    dummy_regressor_results_test_rs[regressorname] = get_regression_metrics(
        regressor, X_test, Y_test,
        rescale=True, rescale_rule='norm_Tn', dataframe=df
    )

### output to screen
print('\n######\n')
# print("Dummy models on training data:\n", dummy_regressor_results_train)
print("Dummy models on test data:\n", dummy_regressor_results_test)
print('\n### Rescaled metrics')
# print("Dummy models on training data:\n", dummy_regressor_results_train_rs)
print("Dummy models on test data:\n", dummy_regressor_results_test_rs)
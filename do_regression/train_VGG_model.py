'''
Example script of how to train one of the VGG models that makes up IcePic
Some key variables:
    - N_ANGLES, N_SHIFTS
            random rotational and translation data augmentation (see methods section of paper)
            sets no. of times done to each image
    - N_EPOCHS
            no. of training epochs employed for training the neural network
    - n_blocks, filters, filter_sizes
            sets architecture of neural network
    - STRAT_BOOL
            whether splitting database via random stratification (Fig 1 of paper)
            or splitting based off of atomic symmetry and/or substrate type (Fig 2 of paper)
Note: below script just has a single example setting for each.
    eg. Amount of data augmentation is quite low, and sow is N_EPOCHS to allow this
    to readily run on a computer without large RAM and walltimes


Location of data:
    - This is set up to work for the database structure of the systems as found in apollo
    - Key parameter is the location of the water contact layers (wat_zpred.txt files in original work)
'''

import numpy as np
import pandas as pd

## sklearn
from sklearn.model_selection import train_test_split

## general code
from general.code import normalise_Tn, make_database_rot_and_shift_rs

## model
from models.regression import model_regress_vgg

###### SOME GLOBAL VARIABLES #######
IM_SIZE = 50 # size of images
N_ANGLES = 2
N_SHIFTS = 2
RANDOM_SEED = np.random.randint(0, high=9999999)
N_EPOCHS = 50

## Read in dataframe
df = pd.read_csv('../dataframe.txt', sep=" ",header=0)

## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

######
## Good or bad nucleator - use for stratifying dataset
THRESHOLD = 210
TARGET = 'Tnuc'
TARGET_BINARY = 'class'
df[TARGET_BINARY] = [1 if value > THRESHOLD else 0 for value in df[TARGET]]
NUM_CLASSES = 2 # important var - used in model building


#### Split into test/train
STRAT_BOOL = True # whether splitting via stratification or not

## if stratifying
if STRAT_BOOL:
    XX = 0.7  # proportional size of training data set
    XY = 0.3
    df_train_stratified, df_test_stratified = train_test_split(df,
                                                               train_size=XX,
                                                               test_size=XY,
                                                               random_state=RANDOM_SEED,
                                                               stratify=df[TARGET_BINARY])
## else spliting based on name of systems
else:
    # must set split string
    # options from paper fig 2:
    # SQ, HE, TR, OC, LJ_fcc100, LJ_fcc111, LJ_fcc110, LJ_fcc211, GR
    SPLIT_STR = 'HE' # example
    trainSYS = []
    testSYS = []
    for i, val in enumerate(df['sysname']):
        if SPLIT_STR in val:
            testSYS.append(i)
        else:
            trainSYS.append(i)
    df_train_stratified = df.iloc[trainSYS]
    df_test_stratified = df.iloc[testSYS]

#### MAKE TRAIN AND TEST DATABASE
X_train, Y_train = make_database_rot_and_shift_rs(df_train_stratified,target='Tnuc_norm',
                                                n_angles = N_ANGLES,
                                                n_shifts = N_SHIFTS,
                                                new_size=(IM_SIZE,IM_SIZE))

X_test, Y_test = make_database_rot_and_shift_rs(df_test_stratified,target='Tnuc_norm',
                                                n_angles = N_ANGLES,
                                                n_shifts = N_SHIFTS,
                                                new_size=(IM_SIZE,IM_SIZE))

##### RESHAPE
print("Shape of training database prior to reshape:",X_train.shape)
X_train = X_train.reshape(X_train.shape[0], IM_SIZE, IM_SIZE, 1)
X_test = X_test.reshape(X_test.shape[0], IM_SIZE, IM_SIZE, 1)
print("Shape of training database after reshape:",X_train.shape)
print("Range of values of labels:", [np.min(Y_train), np.max(Y_train)])

##### TRAIN MODEL
reg_model, history = model_regress_vgg(X_train, Y_train, X_test, Y_test,
                                       IM_SIZE,
                                       N_EPOCHS=N_EPOCHS,
                                       n_blocks=3,
                                       filters=[32,64,128],
                                       filter_sizes=[5, 5, 5],
                                       # set name of model obtained from final epoch
                                       model_file_name_best='mod_final',
                                       # set name of best performing model (on test dataset)
                                       model_file_name_final='mod_best'
                                       )
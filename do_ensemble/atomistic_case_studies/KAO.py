'''
Ensemble prediction on the atomistic AgINaCl setup

Database was randomly split with stratification (see paper) 6 times and ensemble models were created for each split.
Each split is contained in: STRAT1/ STRAT2/ STRAT3/ STRAT4/ STRAT5/ STRAT6/.
Mean of individual neural networks that make up the committees are taken as the prediction.

A final version of IcePic is to combine each of these 6 committee models into one final model.
This is used for predictions on external systems.
The mean of the 6 models is the prediction and the standard deviation between the 6 models is the error bar.

Below gets results for each of the STRATN/ which are then combined (external to this script) to give final predictions
'''
import pandas as pd
from general.code import *
import glob
from keras.models import load_model

####### system making prediction on
zpredFILE = 'wat_zpred.txt'

##### Some global dataset variables
IM_SIZE = 50  # size of images
## Read in dataframe
df = pd.read_csv('../../dataframe.txt', sep=" ", header=0)
## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

###############################################
####### Where to get models for prediction from
for STRAT_N in [1, 2, 3, 4, 5, 6]:
    print('##### STRAT:', STRAT_N, '\n')
    DIR = '../../../CNN_Models/ProductionModel/STRAT{}'.format(STRAT_N)
    # Ensure below is correct path to the model names
    models = glob.glob('{0}/models/mod*'.format(DIR))
    print(np.array(models))

    ## load models now - ready to call later
    models_ld = []
    for m in models:
        models_ld.append(load_model(m))
    print('\n###### Models loaded in:\n', models_ld)

    ###### GET image
    X_test = []
    path = '../../../atomistic_INPs/Kaolinite_TIP4PICE/{0}'.format(zpredFILE)
    # read in zpred and cut to highest density
    dummy_im, _ = make_zpred_squarecut(zpred=read_in_zpred(path), limit=200)
    # normalise - divide by largest value present so [0,1]
    dummy_im = dummy_im / 0.4
    # resize
    dummy_im = resize(dummy_im, (IM_SIZE,IM_SIZE))
    # assign X and Y
    X_test.append(dummy_im) # image
    # no Y_test exists

    X_test = np.array(X_test)
    # reshape
    X_test = X_test.reshape(X_test.shape[0], IM_SIZE, IM_SIZE, 1)

    ### Evaluate models prediction for surfaces
    ypred_ensemble = np.full(X_test.shape[0], 0.0)
    ypred_list = []
    for m in models_ld:
        ypred = m.predict(X_test)
        ypred_list.append(ypred.flatten())
        # sum ensemble predictions
        ypred_ensemble += ypred.flatten()
    # average ensemble prediction
    ypred_ensemble = ypred_ensemble/float(len(models_ld))
    ypred_list = np.array(ypred_list)

    # summary
    print('\n####### Prediction summary')
    print('Individual predictions:\n', convert_normT_2_unnormT(ypred_list, df))
    print('Ensmble prediction:\n', convert_normT_2_unnormT(ypred_ensemble, df))


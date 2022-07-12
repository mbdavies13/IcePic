'''
Script to recreate the Fig. 3 of the paper
'''

import shap
from general.code import *
from tensorflow.keras.models import load_model
import pandas as pd
import time
from general.interpretation import my_shap_plot


# OUTPUT
# graphname = # set to your choice

########################
tf.compat.v1.disable_v2_behavior()

## some global variables
IM_SIZE = 50
N_background = 700 # number of images to take as background for SHAP calculation (more = better)
IM_SIZE = 50  # size of images
## Read in dataframe
df = pd.read_csv('../dataframe.txt', sep=" ", header=0)
## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

#### Stratified splits - select model to employ SHAP upon
STRAT_N =  '2'
DIR = '../../CNN_Models/ProductionModel/STRAT{}'.format(STRAT_N)
STRAT_FILE = 'strat_index.{}.txt'.format(STRAT_N)
my_model = load_model('../../CNN_Models/ProductionModel/STRAT2/models/mod_4_5_5_5_5_32_64_128_256_run7')

#### Split into test/train
f = open('{0}/{1}'.format(DIR, STRAT_FILE)
         , "rb")
train_index, test_index = pickle.load(f)
f.close()
df_train_stratified = df.iloc[train_index]
df_test_stratified = df.iloc[test_index]

#### get test and train database
X_train, Y_train = make_database_simple_w_resize(df_train_stratified,
                                                 target='Tnuc',
                                                 new_size=(IM_SIZE,IM_SIZE),
                                                 zpredFILE='wat_zpred.txt')
X_train = X_train.reshape(X_train.shape[0], IM_SIZE, IM_SIZE, 1)

###### SYSTEMS to employ SHAP values on
# - These systems are in test dataset for this particular stratification
# ... thus testing model predictions and not just what it has been fitted to
list_sys_good = [
    'OH_Ic001_Cfull',
    'LJ_fcc110_a10e9',
    'OH_basal_full_0.05',
    'LJ_fcc111_a2e5'
]

list_sys_bad = [
    'OH_SQU_d3.5_0.20',
    'OH_REC_d2.9_d4.0_0.05',
    'LJ_fcc111_a9e10',
    'LJ_fcc111_a7e10'
]

list_sys = np.concatenate([list_sys_bad, list_sys_good])

####### X_TEST - systems of interest
X_test, Y_test = make_database_simple_w_resize(df[df['sysname'].isin(list_sys)],
                                                 target='Tnuc',
                                                 new_size=(IM_SIZE,IM_SIZE),
                                                 zpredFILE='wat_zpred.txt')
###
# rotate the good square so consistent orientation
sysname = 'OH_Ic001_Cfull'
path = '../../structures/{}/wat_zpred.txt'.format(sysname)
dummy_im = zpred=read_in_zpred(path)
dummy_im = rotate(dummy_im, 45)
dummy_im, _ = make_zpred_squarecut(dummy_im, limit=200)
dummy_im = dummy_im / 0.4
dummy_im = resize(dummy_im, (50,50))
X_test = np.array([val if i < 7 else dummy_im for i,val in enumerate(X_test)])
###
# reshape
X_test = X_test.reshape(X_test.shape[0], IM_SIZE, IM_SIZE, 1)
####################
for image in X_test:
    image_show(image)
    plt.show()


###### SHAP ######
## select a set of background examples to take an expectation over
start_time=time.time()
background = X_train[np.random.choice(X_train.shape[0], N_background, replace=False)]
# background = X_train
print('\n## Setting expectation over {} background database...'.format(background.shape))
e = shap.DeepExplainer(my_model, background)
print('Took {} s'.format(round(time.time()-start_time), 2))


## calc shap values on test database images
# 1 - the images
images  = X_test
# 2 - those images Tn
images_labels = Y_test

# run shap calc
print('\n## Calculating shap over {} test database...'.format(images.shape))
shap_values = e.shap_values(images)
print('Took {} s'.format(round(time.time()-start_time), 2))

# make predictions on test data - so can annotate plot with this
ypred = convert_normT_2_unnormT(
    my_model.predict(images.reshape(len(images), 50, 50, 1)
                     ).flatten(),
                     df
    )

# provide predictions, actual and the images to function
my_shap_plot(images,
             # convert normalised shap values to effect on nulceation temperature in Kelvin
             SHAP_VALUES=convert_normT_2_unnormT(np.array(shap_values), df) - 200,
             Y_ACTUAL=images_labels,
             Y_PRED=ypred,
             shap_global_norm=True,
             wc_global_norm=True,
             cmap_shap='bwr',
             # shift limits of cmap down - more palatable images
             v_shift=-0.2,
             graphname=graphname  # givename for output file here
             )



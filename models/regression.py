import tensorflow as tf  # tensorflow 2.0
from sklearn.metrics import (mean_absolute_error, mean_squared_error)
import numpy as np


## CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout

def add_VGG_block(MODEL, n_filters, filter_size, IM_SIZE, first_block=False):
    from keras.layers.normalization import BatchNormalization

    ## if making first block - provide input shape
    if first_block == True:
        # add first convolution layer
        MODEL.add(Conv2D(n_filters, kernel_size=(filter_size, filter_size),
                         activation='relu',
                         input_shape=(IM_SIZE, IM_SIZE, 1),
                         padding='same',
                         kernel_initializer='he_uniform'
                         ))
    ## dont need to give input if it is a later block
    else:
        MODEL.add(Conv2D(n_filters, (filter_size, filter_size),
                         activation='relu',
                         padding='same',
                         kernel_initializer='he_uniform'
                         ))
    ## then add the rest
    MODEL.add(Conv2D(n_filters, (filter_size, filter_size),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_uniform'
                     ))
    ## add batch normalisation
    MODEL.add(BatchNormalization())

    # add 2D max pooling
    MODEL.add(MaxPooling2D(pool_size=(2, 2)))


def model_regress_vgg(X_train, Y_train, X_test, Y_test, IM_SIZE, N_EPOCHS, loss_function='mean_squared_error',
                      n_blocks=3, filters=[32, 64, 128], filter_sizes=[5, 5, 5],
                      model_file_name_best=None, model_file_name_final=None,
                      verbose=1):
    '''
    CNN model
    X/Y_train = training data
    X/Y_test = test data
    IM_SIZE = size of images
    N_EPOCHS = no. of epochs to run
    loss_function = give name of loss function
    n_blocks = no. of VGG blocks
    filters = list containing no. filters for sequential VGG blocks
    filter_sizes = size of the filters in sequential VGG blocks
    model_file_name_best/final = name of files to store models for best and final model created

    '''
    if len(filters) != n_blocks:
        return ValueError(
            'Filters given as: {0} but ask for {1} VGG blocks. Thus, filters cant be applied.'.format(filters,
                                                                                                      n_blocks))

    model = Sequential()  # add model layers

    ### add VGG blocks ###
    for n in np.arange(n_blocks):
        if n == 0:
            add_VGG_block(model, filters[n], filter_sizes[n], IM_SIZE, first_block=True)
        else:
            add_VGG_block(model, filters[n], filter_sizes[n], IM_SIZE)

    # flatten data - convert from NHWC format to a 2D matrix
    # where rows is batch size, and columns are data features
    model.add(Flatten())

    # add a dense all-to-all relu layer
    model.add(Dense(1024, activation='relu'))

    # apply dropout with rate 0.5
    model.add(Dropout(0.5))

    ## Regression layer
    model.add(Dense(1, activation="linear"))

    ##### COMPILE MODEL
    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss=loss_function, metrics=['mean_absolute_error'])

    ##### see structure of model
    model.summary()

    ##### TRAIN MODEL
    # train the model
    # SAVE: set up to save best model (= one with lowest loss)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(model_file_name_best,
                                                  save_best_only=True,
                                                  monitor='val_loss',
                                                  mode='min')

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                        epochs=N_EPOCHS,
                        callbacks=[mcp_save],
                        verbose=verbose)

    # SAVE: set up to final model (including all weights and compile state)
    model.save(model_file_name_final)

    ##### EVALUATE MODEL
    # evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=1)
    # print performance
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    ##### Predictions
    Y_trainpred = model.predict(X_train)
    Y_testpred = model.predict(X_test)
    print("\nTrain MSE: %.4f" % mean_squared_error(Y_train, Y_trainpred))
    print("Test MSE: %.4f\n" % mean_squared_error(Y_test, Y_testpred))

    ##### SOME EXAMPLE PREDICTIONS
    # predict first 4 images in the test set
    print('\nExample predictions vs. actual')
    print(model.predict(X_test[:4]))
    # Let’s compare this with the actual results.
    # actual results for first 4 images in test set
    print(Y_test[:4])

    return model, history



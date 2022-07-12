import tensorflow as tf  # tensorflow 2.0
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

import random

## skimage
from scipy.ndimage import rotate
from skimage.transform import resize
from keras.utils import plot_model

import colorcet as cc

# model.fit()
# verbose: 'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

# simple function to plot images
def image_show(image, nrows=1, ncols=1,
               show_cmap=False, cmap=cc.cm.kbc, vmin=None, vmax=None,
               figsize=(14,14), log=False):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    p1 = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_cmap:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar with global min max for shap values
        cbar = plt.colorbar(p1, cax=cax)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    print("Image size:", image.shape, image.shape[0]*image.shape[1])
    return fig, ax


def read_in_zpred(path, verbose=False, return_grid=False):
    # reading in python2 pickle into python3 - need latin1 encoding

    with open(os.path.abspath(path), 'rb') as f:
        try:
            zpred,gridx,gridy,conv_factor = pickle.load(f, encoding='latin1')
        except :
            print("Error during unpickling occured - something is wrong with the zpred most likely - just skipping this system")

    if verbose:
        print("zpred read in:", zpred,zpred.shape)
        print("gridx/y read in:",gridx,gridx.shape,"\n",gridy,gridy.shape)
        print("conv_factor read in:", conv_factor)

    # change from 64 to 32 bit float - (halves RAM cost)
    zpred = zpred.astype(np.float32)

    if return_grid:
        return zpred,gridx,gridy
    else:
        return zpred


def plot_train_history(history, accuracy_metric='accuracy', log_bool=False):
    '''
    Plot the history of training
    '''
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 10]
    fig = plt.figure()

    # summarize history for accuracy
    ax = fig.add_subplot(121)
    ax.plot(history.history[accuracy_metric], lw=4)
    ax.plot(history.history['val_{}'.format(accuracy_metric)], lw=4)
    ax.set_title('model {}'.format(accuracy_metric))
    ax.set_ylabel('value')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    if log_bool == True:
        ax.set_yscale('log')

    # summarize history for loss
    ax = fig.add_subplot(122)
    ax.plot(history.history['loss'], lw=4)
    ax.plot(history.history['val_loss'], lw=4)
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    if log_bool == True:
        ax.set_yscale('log')

    plt.tight_layout()

    return fig


def plot_regression_fit(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, rescale_bool=False, rescale_rule='norm',
                        dataframe=None):
    '''
    Plot regression fit

    rescale_rule = how it was scaled. norm = normalised, cbrt = cube root
    '''

    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 30]
    fig = plt.figure()

    ### Train
    ax = fig.add_subplot(411)
    ylearn = MODEL.predict(X_TRAIN)
    x_ax = range(len(ylearn))
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(x_ax, convert_normT_2_unnormT(Y_TRAIN, dataframe), s=50, color="blue", label="actual")
            ax.plot(x_ax, convert_normT_2_unnormT(ylearn, dataframe), lw=2, color="red", label="learned")
        elif rescale_rule == 'cbrt':
            ax.scatter(x_ax, Y_TRAIN ** 3, s=50, color="blue", label="actual")
            ax.plot(x_ax, ylearn ** 3, lw=2, color="red", label="learned")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    elif rescale_bool == False:
        ax.scatter(x_ax, Y_TRAIN, s=50, color="blue", label="actual")
        ax.plot(x_ax, ylearn, lw=2, color="red", label="learned")
    ax.legend()

    ax = fig.add_subplot(412)
    ylearn = MODEL.predict(X_TRAIN)
    x_ax = range(len(ylearn))
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(x_ax, convert_normT_2_unnormT(Y_TRAIN, dataframe) - convert_normT_2_unnormT(ylearn.flatten(),
                                                                                                   dataframe),
                       s=50, color="blue", label="train residuals")
        elif rescale_rule == 'cbrt':
            ax.scatter(x_ax, (Y_TRAIN ** 3) - (ylearn.flatten() ** 3),
                       s=50, color="blue", label="train residuals")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    elif rescale_bool == False:
        ax.scatter(x_ax, Y_TRAIN - ylearn.flatten(),
                   s=50, color="blue", label="train residuals")
    ax.legend()

    ### Test
    ax = fig.add_subplot(413)
    ypred = MODEL.predict(X_TEST)
    x_ax = range(len(ypred))
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(x_ax, convert_normT_2_unnormT(Y_TEST, dataframe), s=50, color="blue", label="actual")
            ax.plot(x_ax, convert_normT_2_unnormT(ypred, dataframe), lw=2, color="red", label="predicted")
        elif rescale_rule == 'cbrt':
            ax.scatter(x_ax, Y_TEST ** 3, s=50, color="blue", label="actual")
            ax.plot(x_ax, ypred ** 3, lw=2, color="red", label="predicted")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    elif rescale_bool == False:
        ax.scatter(x_ax, Y_TEST, s=50, color="blue", label="actual")
        ax.plot(x_ax, ypred, lw=2, color="red", label="predicted")
    ax.legend()

    ax = fig.add_subplot(414)
    ypred = MODEL.predict(X_TEST)
    x_ax = range(len(ypred))
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(x_ax,
                       convert_normT_2_unnormT(Y_TEST, dataframe) - convert_normT_2_unnormT(ypred.flatten(), dataframe),
                       s=50, color="blue", label="test residuals")
        elif rescale_rule == 'cbrt':
            ax.scatter(x_ax, (Y_TEST ** 3 - ypred.flatten() ** 3),
                       s=50, color="blue", label="test residuals")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    elif rescale_bool == False:
        ax.scatter(x_ax, Y_TEST - ypred.flatten(),
                   s=50, color="blue", label="test residuals")
    ax.legend()

    plt.tight_layout()

    return fig

def plot_ensemble_regression_fit(members, X_TEST, Y_TEST, rescale_bool=False,
                        dataframe=None, preloaded=False):
    '''
    Plot regression fit of an aensmble of models - along with the mean result.

    rescale_rule = how it was scaled. norm = normalised, cbrt = cube root
    '''
    from keras.models import load_model
    import gc

    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 15]
    fig = plt.figure()

    x_ax = range(len(Y_TEST))

    ### Test
    ax = fig.add_subplot(211)
    ypred_list=[]
    for MODEL in members:
        if preloaded == False:
            MODEL = load_model(MODEL)
        ypred = MODEL.predict(X_TEST).flatten()
        ypred_list.append(ypred)
        # ax.plot(x_ax, ypred,
        #         lw=2, color="red", alpha=0.75)

        # deal with multiple prediction memory leak
        _ = gc.collect()

    ax.scatter(x_ax, Y_TEST, s=50, color="blue", label="actual")
    ax.plot(x_ax, np.mean(ypred_list, axis=0),
            lw=2, color="red", label="predicted")
    ax.fill_between(x_ax, np.min(ypred_list, axis=0), np.max(ypred_list, axis=0),
                    lw=2, color="red", alpha=0.25)
    ax.legend(framealpha=0.25)

    ### Test residuals
    ax = fig.add_subplot(212)
    ypred_list=[]
    for MODEL in members:
        if preloaded == False:
            MODEL = load_model(MODEL)
        ypred = MODEL.predict(X_TEST).flatten()
        ypred_list.append(ypred)
        # ax.plot(x_ax, Y_TEST - ypred,
        #         lw=2, color="b", alpha=0.75)

        # deal with multiple prediction memory leak
        _ = gc.collect()

    ax.plot(x_ax, Y_TEST - np.mean(ypred_list, axis=0),
               lw=2, color="blue")
    ax.scatter(x_ax, Y_TEST - np.mean(ypred_list, axis=0),
               s=50, color="blue", label="test residuals")
    ax.fill_between(x_ax, Y_TEST - np.min(ypred_list, axis=0), Y_TEST - np.max(ypred_list, axis=0),
                    lw=2, color="blue", alpha=0.25)
    ax.legend(framealpha=0.25)

    plt.tight_layout()

    return fig

def plot_classification_fit(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    '''
    Plot regression fit
    '''

    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 30]
    fig = plt.figure()

    ### Train
    ax = fig.add_subplot(411)
    ax.axhline(0.5, linestyle='dashed', color='k', lw=3)
    ylearn = MODEL.predict(X_TRAIN)
    x_ax = range(len(ylearn))
    ax.scatter(x_ax, Y_TRAIN[:, 0], s=50, marker='x', color="blue", label="actual 0")
    ax.plot(x_ax, ylearn[:, 0], lw=2, color="red", label="learned 0")
    ax.legend()

    ax = fig.add_subplot(412)
    ylearn = MODEL.predict(X_TRAIN)
    x_ax = range(len(ylearn))
    ax.scatter(x_ax, Y_TRAIN[:, 0] - ylearn[:, 0], s=50, color="blue", label="train residuals")
    ax.legend()

    ### Test
    ax = fig.add_subplot(413)
    ax.axhline(0.5, linestyle='dashed', color='k', lw=3)
    ypred = MODEL.predict(X_TEST)
    x_ax = range(len(ypred))
    ax.scatter(x_ax, Y_TEST[:, 0], s=50, color="blue", label="actual 0 ")
    ax.plot(x_ax, ypred[:, 0], lw=2, color="red", label="predicted 0 ")
    ax.legend()

    ax = fig.add_subplot(414)
    ypred = MODEL.predict(X_TEST)
    x_ax = range(len(ypred))
    ax.scatter(x_ax, Y_TEST[:, 1] - ypred[:, 1], s=50, color="blue", label="test residuals")
    ax.legend()

    plt.tight_layout()

    return fig


def plot_regression_residualPDF(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, rescale_bool=False, rescale_rule='norm',
                                dataframe=None):
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 10]

    ## MAE
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ypred = MODEL.predict(X_TEST)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            pdf, bin_edges = np.histogram(
                convert_normT_2_unnormT(Y_TEST, dataframe) - convert_normT_2_unnormT(ypred.flatten(), dataframe),
                density=True)
            MAE = np.mean(np.abs(
                convert_normT_2_unnormT(Y_TEST, dataframe) - convert_normT_2_unnormT(ypred.flatten(), dataframe)))
        elif rescale_rule == 'cbrt':
            pdf, bin_edges = np.histogram(Y_TEST ** 3 - ypred.flatten() ** 3, density=True)
            MAE = np.mean(np.abs(Y_TEST ** 3 - ypred.flatten() ** 3))
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        pdf, bin_edges = np.histogram(Y_TEST - ypred.flatten(), density=True)
        MAE = np.mean(np.abs(Y_TEST - ypred.flatten()))
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.plot(bincenters, pdf, lw=5, c='b')
    ax.axvline(MAE, c='k', lw=5, linestyle='dashed', label='MAE: {}'.format(round(MAE, 6)))
    ax.set_ylabel('PDF')
    ax.set_xlabel('actual - predicted')
    ax.legend()

    ## MSE
    ax = fig.add_subplot(122)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            SE = (convert_normT_2_unnormT(Y_TEST, dataframe) - convert_normT_2_unnormT(ypred.flatten(), dataframe)) ** 2
            MSE = np.mean(SE)
        elif rescale_rule == 'cbrt':
            SE = (Y_TEST ** 3 - ypred.flatten() ** 3) ** 2
            MSE = np.mean((Y_TEST ** 3 - ypred.flatten() ** 3) ** 2)
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        SE = (Y_TEST - ypred.flatten()) ** 2
        MSE = np.mean((Y_TEST - ypred.flatten()) ** 2)

    pdf, bin_edges = np.histogram(SE, density=True)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.plot(bincenters, pdf, lw=5, c='b')
    RMSE = MSE ** 0.5
    ax.axvline(MSE, c='k', lw=5, linestyle='dashed', label='MSE: {0}'.format(round(MSE, 6)))
    ax.axvline(RMSE, c='k', lw=5, linestyle='dotted', label='RMSE: {0}'.format(round(RMSE, 6)))
    ax.set_ylabel('PDF')
    ax.set_xlabel('(actual - predicted)**2')
    ax.legend()

    plt.tight_layout()

    return fig


def plot_residuals_vs_target(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, rescale_bool=False, rescale_rule='norm',
                             dataframe=None):
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 15]
    fig = plt.figure()

    ### Train
    ax = fig.add_subplot(121)
    ylearn = MODEL.predict(X_TRAIN)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            r = convert_normT_2_unnormT(Y_TRAIN, dataframe) - convert_normT_2_unnormT(ylearn.flatten(), dataframe)
            ax.scatter(convert_normT_2_unnormT(Y_TRAIN, dataframe), r, s=50, color="blue", label="train residuals")
        elif rescale_rule == 'cbrt':
            r = Y_TRAIN ** 3 - ylearn.flatten() ** 3  # residuals
            ax.scatter(Y_TRAIN**3, r, s=50, color="blue", label="train residuals")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        r = Y_TRAIN - ylearn.flatten()  # residuals
        ax.scatter(Y_TRAIN, r, s=50, color="blue", label="train residuals")
    ax.set_xlabel('actual value')
    ax.set_ylabel('redisual')
    ax.legend()

    ### Test
    ax = fig.add_subplot(122)
    ypred = MODEL.predict(X_TEST)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            r = convert_normT_2_unnormT(Y_TEST, dataframe) - convert_normT_2_unnormT(ypred.flatten(), dataframe)
            ax.scatter(convert_normT_2_unnormT(Y_TEST, dataframe), r, s=50, color="blue", label="test residuals")
        elif rescale_rule == 'cbrt':
            r = Y_TEST ** 3 - ypred.flatten() ** 3  # residuals
            ax.scatter(Y_TEST**3, r, s=50, color="blue", label="test residuals")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        r = Y_TEST - ypred.flatten()  # residuals
        ax.scatter(Y_TEST, r, s=50, color="blue", label="test residuals")
    ax.set_xlabel('actual value')
    ax.set_ylabel('redisual')
    ax.legend()

    plt.tight_layout()

    return fig


def plot_regression_vs_target(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, rescale_bool=False, rescale_rule='norm',
                              dataframe=None):
    from sklearn.metrics import r2_score

    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 15]
    fig = plt.figure()

    ### Train
    ax = fig.add_subplot(121)
    ylearn = MODEL.predict(X_TRAIN)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(convert_normT_2_unnormT(Y_TRAIN, dataframe),
                       convert_normT_2_unnormT(ylearn.flatten(), dataframe), s=50, color="blue", label="train")
        elif rescale_rule == 'cbrt':
            ax.scatter(Y_TRAIN ** 3, ylearn.flatten() ** 3, s=50, color="blue", label="train")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        ax.scatter(Y_TRAIN, ylearn.flatten(), s=50, color="blue", label="train")
    ax.set_xlabel('label actual value')
    ax.set_ylabel('learnt value')
    ax.legend()
    ax.set_title('R2: {:.3g}'.format(r2_score(Y_TRAIN, ylearn)))
    # y = x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, lw=4)

    ### Test
    ax = fig.add_subplot(122)
    ypred = MODEL.predict(X_TEST)
    if rescale_bool == True:
        if rescale_rule == 'norm':
            ax.scatter(convert_normT_2_unnormT(Y_TEST, dataframe), convert_normT_2_unnormT(ypred.flatten(), dataframe),
                       s=50, color="blue", label="test")
        elif rescale_rule == 'cbrt':
            ax.scatter(Y_TEST ** 3, ypred.flatten() ** 3, s=50, color="blue", label="test")
        else:
            raise ValueError('Don\'t understand rescale rule {}'.format(rescale_rule))
    else:
        ax.scatter(Y_TEST, ypred.flatten(), s=50, color="blue", label="test")
    ax.set_xlabel('label actual value')
    ax.set_ylabel('predicted value')
    ax.legend()
    ax.set_title('R2: {:.3g}'.format(r2_score(Y_TEST, ypred)))
    # y = x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, lw=4)

    plt.tight_layout()

    return fig

def plot_ensemble_regression_vs_target(members, X_TEST, Y_TEST, rescale_bool=False, rescale_rule='norm',
                              dataframe=None, preloaded=False):
    from keras.models import load_model
    import gc
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [15, 15]
    fig = plt.figure()

    ### Test
    ax = fig.add_subplot(111)

    ypred_ensemble = np.full(Y_TEST.flatten().shape, 0.0)
    for MODEL in members:
        if preloaded == False:
            MODEL = load_model(MODEL)
        if rescale_bool == True:
            if rescale_rule == 'norm':
                ypred = convert_normT_2_unnormT(MODEL.predict(X_TEST).flatten(), dataframe)
            elif rescale_rule == 'cbrt':
                ypred = MODEL.predict(X_TEST).flatten() ** 3.0
        else:
            ypred = MODEL.predict(X_TEST).flatten()
        # sum ensemble predictions
        ypred_ensemble += ypred

        # deal with multiple prediction memory leak
        _ = gc.collect()
    # average ensemble prediction
    ypred_ensemble = ypred_ensemble/float(len(members))

    # rescale actual values
    if rescale_bool and rescale_rule == 'norm':
        Y_TEST = convert_normT_2_unnormT(Y_TEST.flatten(), dataframe)
    elif rescale_bool and rescale_rule == 'norm':
        Y_TEST = Y_TEST ** 3.0

    ax.scatter(Y_TEST, ypred_ensemble, s=200, lw=5, c="cyan", ec='blue', alpha=0.8, label="test")
    ax.set_xlabel('actual value')
    ax.set_ylabel('predicted value')
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()

    # y = x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, lw=4)

    return fig


def plot_class_residuals_vs_target(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, Y_OGTRAIN, Y_OGTEST):
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [30, 15]
    fig = plt.figure()

    ### Train
    ax = fig.add_subplot(121)
    ylearn = MODEL.predict(X_TRAIN)
    r = Y_TRAIN[:, 0] - ylearn[:, 0]  # residuals
    ax.scatter(Y_OGTRAIN, r, s=50, color="blue", label="train residuals for 0")
    ax.set_xlabel('label actual value')
    ax.set_ylabel('actual - learned')
    ax.legend()

    ### Test
    ax = fig.add_subplot(122)
    ypred = MODEL.predict(X_TEST)
    r = Y_TEST[:, 0] - ypred[:, 0]  # residuals
    ax.scatter(Y_OGTEST, r, s=50, color="blue", label="test residuals for 0")
    ax.set_xlabel('label actual value')
    ax.set_ylabel('actual - predicted')
    ax.legend()

    plt.tight_layout()

    return fig

def OLD_make_zpred_squarecut(zpred,limit=200):
    """########ZPRED SQUARE CUT#######
    ###cutting zpred to be square with largest possible density enclosed###
    ###AND to have max size of "limit"
    Limit: 100 = 1 nm
    Inputs: zpred - the gaussian surface to make square cut from"""
    ##square dimensions:
    if np.min(zpred.shape) < limit:
        limited = False
        raise ValueError('Full Image is smaller than size want to make cut so image cannot be made into correct data size - FIX!')
        #then smallest side of zpred rectangle is less than limit so have square dim based off of that
        #square_dim=np.min(zpred.shape) #equal smallest side of zpred rectangle
        ##extract which dimension was the shortest
        #x_or_y=np.argmin(zpred.shape) #0 = x, 1 = y
        #limited=False
    else: #set square size myself
        #print "square limited"
        square_dim=limit #equal smallest side of zpred rectangle
        #extract which dimension was the shortest
        x_or_y=np.argmin(zpred.shape) #0 = x, 1 = y
        limited=True
    #Now move square along grid and find square of highest dimension#
    #making list of positions to move square (only 1d movement as maxed out on one dimension)
    s1,s2 = zpred.shape
    t1,t2 = (square_dim,square_dim)
    ex_max=s1-t1
    ey_max=s2-t2
    if x_or_y == 0: #if x is the smallest dimension
        if limited == False:
            #square fills one dimension - so just move along the other
            #print "x dimension was shortest of zpred rectangle - hence taking as maximum of square cut dimension.
            #Move the suqare along y direction to find max density location."
            roll_list=zip((np.zeros(len(np.arange(ey_max)),dtype='int')),np.arange(ey_max))
        if limited == True:
            #square does not fill either dimension - want to set it to move half way along the large dimension at half way up the smaller dimension
            xroll=np.array((np.zeros(len(np.arange(ey_max)))+int(ex_max/2)),dtype='int')
            roll_list=zip(xroll,np.arange(ey_max))
    elif x_or_y == 1: #y is smallest direction
        if limited == False:
            #print "y dimension was shortest of zpred rectangle - hence taking as maximum of square cut dimension. Move the square along x direction to find max density location."
            roll_list=zip(np.arange(ex_max),(np.zeros(len(np.arange(ex_max)),dtype='int')))
        if limited == True:
            yroll=np.array(np.zeros(len(np.arange(ex_max)))+int(ey_max/2),dtype='int')
            roll_list=zip(np.arange(int(ex_max/2),dtype='int'),yroll)
    #        print roll_list
    else:
        raise ValueError('***ERROR***x_or_y not determined')

    ##now move the square along and take the maximum density position
    density_list=[]
    for i,val in enumerate(roll_list):
    #    print i,val
        edge_coordinate = (val[0],val[1])
        slicer = tuple(slice(edge, edge+i) for edge, i in zip(edge_coordinate, (square_dim,square_dim)))
        density_list.append(np.sum(zpred[slicer]))
    #find max position
    max_position=np.argmax(density_list)
    density_list=[] #remove the unecessary data list (for centurion)
    #make the actual max placement tuple
    if x_or_y == 0:
        if limited == False:
            #print "Found max to be at ({0},{1})".format(0,max_position)
            max_tuple=(0,max_position)
        if limited == True:
            max_tuple=(int(ex_max/2),max_position)
    elif x_or_y == 1:
        if limited == False:
            #print "Found max to be at ({0},{1})".format(max_position,0)
            max_tuple=(max_position,0)
        if limited == True:
            max_tuple=(max_position,int(ey_max/2))
    else:
        raise ValueError("***ERROR***x_or_y not determined")
    #make the slice for the max placement
    slicer = tuple(slice(edge, edge+i) for edge, i in zip(max_tuple, (square_dim,square_dim)))


    return zpred[slicer]

def make_zpred_squarecut(zpred,limit=200):
    """########ZPRED SQUARE CUT#######
    ###cutting zpred to be square with largest possible density enclosed###
    ###AND to have max size of "limit"
    Limit: 100 = 1 nm
    Inputs: zpred - the gaussian surface to make square cut from"""
    ##square dimensions:
    if np.min(zpred.shape) < limit:
        raise ValueError('Full Image is smaller than size want to make cut so image cannot be made into correct data size - FIX!')
    else: #set square size myself
        #print "square limited"
        square_dim=limit #equal smallest side of zpred rectangle
        #extract which dimension was the shortest
        x_or_y=np.argmin(zpred.shape) #0 = x, 1 = y
        limited=True
    #Now move square along grid and find square of highest dimension#
    #making list of positions to move square (only 1d movement as maxed out on one dimension)
    s1,s2 = zpred.shape
    t1,t2 = (square_dim,square_dim)
    ex_max=s1-t1
    ey_max=s2-t2
    if x_or_y == 0: #if x is the smallest dimension
        if limited == False:
            #square fills one dimension - so just move along the other
            #Move the suqare along y direction to find max density location."
            roll_list=zip((np.zeros(len(np.arange(ey_max)),dtype='int')),np.arange(ey_max))
        if limited == True:
            #square does not fill either dimension - want to set it to move half way along the large dimension at half way up the smaller dimension
            xroll=np.array((np.zeros(len(np.arange(ey_max)))+int(ex_max/2)),dtype='int')
            roll_list=zip(xroll,np.arange(ey_max))
    elif x_or_y == 1: #y is smallest direction
        if limited == False:
            roll_list=zip(np.arange(ex_max),(np.zeros(len(np.arange(ex_max)),dtype='int')))
        if limited == True:
            yroll=np.array(np.zeros(len(np.arange(ex_max)))+int(ey_max/2),dtype='int')
            roll_list=zip(np.arange(int(ex_max/2),dtype='int'),yroll)

    else:
        raise ValueError('***ERROR***x_or_y not determined')

    ##now move the square along and take the maximum density position
    density_list=[]
    for i,val in enumerate(roll_list):
    #    print i,val
        edge_coordinate = (val[0],val[1])
        slicer = tuple(slice(edge, edge+i) for edge, i in zip(edge_coordinate, (square_dim,square_dim)))
        density_list.append(np.sum(zpred[slicer]))
    #find max position
    max_position=np.argmax(density_list)
    #make the actual max placement tuple
    if x_or_y == 0:
        if limited == False:
            #print "Found max to be at ({0},{1})".format(0,max_position)
            max_tuple=(0,max_position)
        if limited == True:
            max_tuple=(int(ex_max/2),max_position)
    elif x_or_y == 1:
        if limited == False:
            #print "Found max to be at ({0},{1})".format(max_position,0)
            max_tuple=(max_position,0)
        if limited == True:
            max_tuple=(max_position,int(ey_max/2))
    else:
        raise ValueError("***ERROR***x_or_y not determined")
    #make the slice for the max placement
    slicer = tuple(slice(edge, edge+i) for edge, i in zip(max_tuple, (square_dim,square_dim)))

    return zpred[slicer], slicer

def randomly_split_int_in_two(number, verbose=False):
    ## randomly share the "amplitude_shift" points between x and y
    # method: use np.random.multinomial
    # this sets up a probability test (.e.g rolling dice)
    # I've set this up such that it is like throwing a coin (0 or 1) and doing it
    #.. number times. However, I randomly assign the coins prob of 0 or 1
    # via "p1". Then sum up the total 0s (shifts in x) and 1s (shifts in y) for all the coin tosses
    p1 = random.uniform(0, 1)
    test = np.random.multinomial(n=1, size=number, pvals=[p1, 1.0-p1])
    if verbose == True:
        print('split 1: {0}, split 2: {1}, total: {2}'.format(np.sum(test[:,0]), np.sum(test[:,1]), np.sum(test[:,0])+np.sum(test[:,1])))
    return np.sum(test[:,0]), np.sum(test[:,1])

def positive_or_negative():
    return (-1)**random.randrange(2)

def attempt_shift(attempts, min_shift_size, max_shift_size, d_threshold, ex_max, ey_max, slicer, zpred, zpred_square, limit, verbose):
    """
    Function to attempt a shift. Called by shift_zpred_squarecut()
    It runs a while loop, with two conditionals:
    1 = successfuly shift has occured
    2 = limit of number of iterations

    To be "successful" the shift has three parameters:
    1 = image fits or not - non-negotiable conditional
    2 = change in density less than d_threshold
    3 = a parameter that sets the random size of the shift

    (2) and (3) can thus be softened if while loop fails and this function recalled
    to try again.

    INPUTS:
    attempts = max no. of times attempt shift
    min/max_shfit size = range to draw amplitude of shift from
    d_threshold = percentage drop in density allowed
    ex/ey_max = max edge positions of slicer
    slicer = the original slicer from make_zpred_squarecut()
    zpred = full image
    zpred_square = zpred square cut from make_zpred_squarecut()

    """
    ## finding the available axis - can shift be made
    # ex_max and ey_max are the maximum edge coordinate positions can have
    # i.e. (ex_max,ey_max) is the furthest position from the origin you can make the square cut
    # so we just need to ensure it is smaller than that - can do this with a while loop
    slice_success=False
    counter=0
    while slice_success==False and counter<attempts:

        # randomly assign size of the shift [100 = 1nm]
        amplitude_of_shift = np.random.randint(low=min_shift_size, high=max_shift_size+1)


        ## randomly share the "amplitude_shift" points between x and y
        shift_x, shift_y = randomly_split_int_in_two(amplitude_of_shift, verbose=verbose)

        ## randomly assign sign of the shift
        shift_x = positive_or_negative()*shift_x
        shift_y = positive_or_negative()*shift_y

        # create new slice
        n_s1 = slice(slicer[0].start+shift_x, slicer[0].stop+shift_x, slicer[0].step)
        n_s2 = slice(slicer[1].start+shift_y, slicer[1].stop+shift_y, slicer[1].step)
        new_slicer = (n_s1,n_s2)


        # Evaluate change in density
        d_square = np.sum(zpred_square)
        d_shift = np.sum(zpred[new_slicer])
        d_diff = (d_square - d_shift)/d_square # percentage diff between two

        if verbose == True:
            #print("shifting image by", amplitude_of_shift)
            #print("New slicer:", new_slicer)
            print("Conditionals", d_diff, new_slicer[0].stop-ex_max, new_slicer[1].stop-ey_max)

        # conditionals to be success
        # 1 - if new slice fits in image
        if new_slicer[0].start <= ex_max and new_slicer[1].start <= ey_max:

            # 2 - do density test - ensure not getting empty part of image
            if d_diff < d_threshold:
                slice_success=True

        counter+=1

    ## return new slicer (=the shifted one) if successful
    if slice_success == True:

        ## Final sanity check on image size that taken as final result
        if zpred[new_slicer].shape != (limit,limit):
            raise ValueError('Shifted slice is {} which is not same size as IM_SIZE. Failed to make image appropriate size'.format(zpred[new_slicer].shape))
        else:
            return new_slicer

def shift_zpred_squarecut(zpred, limit=200, verbose=False, return_slicer=False):
    '''
    Function to take zpred - make square cut, and then randomly shift away
    Just uses make_zpred_squarecut() to make the square cut
    Then randomly shifts slicer with random direction and emplitude in [max/min_shiftsize]
    Then checks density drop is not more than d_threshold - to ensure not moving to an empty part of
    the image -> empty sections of images would be problematic!

    INPUTS:
    zpred = FULL zpred of system (not an already squarecut one)
    max/min_shift_size = max and min of random number range for shifting the image
    (100 = 1nm)
    limit = image size (usually 200)

    ARG
    return_slicer - option to return sliceer
    '''

    ## 0 - get the original slicer of max density from squarecut()
    zpred_square, slicer = make_zpred_squarecut(zpred)

    ## find limits of where can place square of size (limit,limit) on grid
    s1, s2 = zpred.shape
    t1, t2 = (limit, limit)
    ex_max = s1 - t1
    ey_max = s2 - t2

    ## For loop - to attempt shift in N different ways - just reducing magnitude
    maxmin_shift_list = [(50, 150), (10, 100), (10, 50)]
    for N in np.arange(3):

        ## Attempt random shift N - try "attempts" no. of times
        m1, m2 = maxmin_shift_list[N]
        new_slicer = attempt_shift(attempts=10000, min_shift_size=m1, max_shift_size=m2,
                                   d_threshold=0.025, ex_max=ex_max, ey_max=ey_max, slicer=slicer,
                                   zpred=zpred, zpred_square=zpred_square, limit=limit, verbose=verbose)

        ## If successful (i.e. new_slicer has been return and exists - then stop)
        if new_slicer != None:
            break

    if new_slicer is None:
        return None
    elif return_slicer == True:
        return zpred[new_slicer], new_slicer
    else:
        return zpred[new_slicer]

def normalise_Tn(data):
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data

def unnormalise_Tn(norm_data, og_data):
    '''Unnormalise data
    Inputs:
    norm_data = normalised data
    og_data = original data
    Output:
    unnorm_data which should equal og_data'''
    unnorm_data = (
        (norm_data * (np.max(og_data) - np.min(og_data)))
               + np.min(og_data)
           )
    return unnorm_data


def mae_from_maenorm_Tn(MAEvalue,dataframe):
    '''
    Return the MAE for actual data from that obtained from normalised.
    I normalise via: x --> (x-c)/d
    Where:
    c = np.min() --> a translational shift to get them from 0 to ~75
    d = --> a scaling to get them from 0 to 1

    MAE = d*MAEnorm (easy to check by hand - translation cancels out)

    '''
    return MAEvalue * (np.max(dataframe['Tnuc']) - np.min(dataframe['Tnuc']))

def rmse_from_msenorm_Tn(MSEvalue, dataframe):
    '''
    Return the RMSE for actual data from the MSE obtained from normalised.
    I normalise via: x --> (x-c)/d
    Where:
    c = np.min() --> a translational shift to get them from 0 to ~75
    d = --> a scaling to get them from 0 to 1

    MSE = d^2*MSEnorm (easy to check by hand - translation cancels out)
    RMSE = sqrt(MSE) = d*sqrt(MSEnorm)

    NOTE: Input is MSEvalue, not the RMSEvalue!!

    '''
    return (np.max(dataframe['Tnuc']) - np.min(dataframe['Tnuc'])) * np.sqrt(MSEvalue)


def make_database_simple_w_resize(dataframe, target, new_size: tuple, im_size=200, zpredFILE='wat_zpred.txt'):
    '''
    Provide a dataframe and return image database with
    each entry labelled by 'target' value

    Loop through each system in dataframe provided
     - take its zpred, and target value

    ARGS:
    dataframe = panda dataframe
    target = name of target column
    im_size = size of image cut taken (using highest density region method)

    '''

    X = []
    Y = []
    e = 0  # error count

    for i in np.arange(len(dataframe)):
        sys.stdout.write("\rSystem: " + str(i) + "/" + str(len(dataframe) - 1))
        sys.stdout.flush()

        path = '../../structures/{0}/{1}'.format(
            np.array(dataframe['sysname'])[i], zpredFILE)

        try:
            # read in zpred and cut to highest density
            dummy_im, _ = make_zpred_squarecut(zpred=read_in_zpred(path), limit=im_size)

            # normalise - divide by largest value present so [0,1]
            dummy_im = dummy_im / 0.4

            # resize
            dummy_im = resize(dummy_im, new_size)

            # assign X and Y
            X.append(dummy_im)  # image
            Y.append(np.array(dataframe[target])[i])  # target label

        # count the errors - result from zpred not being complete etc.
        except:
            e += 1

    X, Y = np.array(X), np.array(Y)

    print("\nPercentage of systems with error and not read in", e / len(dataframe) * 100, "%")
    print("Shape of X database:", X.shape)
    print("Shape of Y database:", Y.shape)

    return X, Y

def make_database_rot_and_shift_rs(dataframe, target, n_angles, n_shifts, new_size: tuple, im_size=200,
                                   zpredFILE='wat_zpred.txt'):
    '''
    Provide a dataframe and return image database with
    each entry labelled by 'target' value

    Loop through each system in dataframe provided
     - take its zpred, and target value
     - data augmentation via rotation and translation
    (note: these ARE different - e.g. a system with ciruclar symmetry - rots will
     do little, whilst translations will do much more - I have seen examples of this!)


    ARGS:
    dataframe = panda dataframe
    target = name of target column
    angels = angles to rotate by
    n_shifts = number of shifts (translations) to make
    im_size = size of image cut taken (using highest density region method)
    zpredFILE = name of water contact layer files

    # NOTE: if dont have default database structure then need to change the path variable!

    '''

    X = []
    Y = []
    e = 0  # error count - from initial reading in of ZPRED
    e_shift = 0  # shift error count

    for i in np.arange(len(dataframe)):
        sys.stdout.write("\rSystem: " + str(i) + "/" + str(len(dataframe) - 1))
        sys.stdout.flush()

        # make random angles array. First entry = 0 such that non rotated is given. Rest are all random
        angles = np.concatenate(([0],np.random.randint(low=1,high=359, size=n_angles-1)), axis=0)

        # KEY VARIABLE - path to water contact layer image
        path = '../../structures/{0}/{1}'.format(
            np.array(dataframe['sysname'])[i], zpredFILE)

        if angles[0] != 0:
            raise ValueError(
                'First rotation angle is not 0 --> won\'t have unshifted original image in training. Correct this.')

        try:
            # 0 - read in zpred and store - NEVER update - need the original and
            # will save time if dont have to read in each iteration of the for loops
            ZPRED = read_in_zpred(path)

            # cut to highest density
            dummy_im, _ = make_zpred_squarecut(ZPRED, limit=im_size)

            # normalise - divide by height of single gaussian
            dummy_im = dummy_im / 0.4

            # resize
            dummy_im = resize(dummy_im, new_size)

            # assign X and Y
            X.append(dummy_im)  # image
            Y.append(np.array(dataframe[target])[i])  # target label

            #### 1 - ROTATE
            for a in angles:
                # 1 - make rotation on FULL image and store as dummy_im_ROT
                # NEVER update - as need for the shifts
                dummy_im_ROT = rotate(ZPRED, angle=a, order=3, reshape=False)
                # 2 - make square cut (must do to full rotated image, not to square cut)
                dummy_im, _ = make_zpred_squarecut(dummy_im_ROT, limit=im_size)

                # 3 - normalise - divide by height of single gaussian
                dummy_im = dummy_im / 0.4

                # 4 - resize
                dummy_im = resize(dummy_im, new_size)

                # assign X and Y
                # if a=0 then dont as this rotation would be duplicate (need dummy though for shifts)
                if a == 0:
                    pass
                else:
                    X.append(dummy_im)
                    Y.append(np.array(dataframe[target])[i])

                #### 2 - SHIFT
                for n in np.arange(n_shifts):
                    # 1 - give the stored full image rotation
                    # make square cut, then randomly shift
                    # shift_zpred_squarecut() calls make_zpred_squarecut so dont need to do that again
                    dummy_im = shift_zpred_squarecut(dummy_im_ROT)

                    # CHANCE THAT SHIFTING FAILS (small but possible)
                    # If that occurs, then nothing is returned.
                    # So only want to continue appending etc. if all works
                    if dummy_im is None:
                        e_shift += 1
                    else:
                        # 3 - normalise - divide by height of single gaussian
                        dummy_im = dummy_im / 0.4

                        # 4 - resize
                        dummy_im = resize(dummy_im, new_size)

                        # assign X and Y
                        X.append(dummy_im)
                        Y.append(np.array(dataframe[target])[i])

        # count the errors - result from zpred not being unpickled etc.
        except:
            e += 1
            print(path)

    X, Y = np.array(X), np.array(Y)

    print("\nPercentage of systems with error and not read in", e / len(dataframe) * 100, "%")
    print("Number of shifts that failed", e_shift)
    print("Shape of X database:", X.shape)
    print("Shape of Y database:", Y.shape)
    print("Min and Max of X:", np.min(X.flatten()), np.max(X.flatten()))
    print('Expected size of database:', len(dataframe) * (1 + len(angles) + (len(angles) * n_shifts) - 1))
    # Expected size = no.systems * (1 (= initial) + no. angles (=rotations done) + no.angles*n_shifts (=shifts for each rotation) - 1 (= angle zero not appended but shifts are))

    return X, Y

def convert_normT_2_unnormT(in_data, dataframe):
    '''
    1 - rescale based off of dataframe
    2 - translated based off of dataframe
    --> unnormalise any data to be in the original scale
    '''
    rescale = (np.max(dataframe['Tnuc']) - np.min(dataframe['Tnuc'])) * np.array(in_data)
    translate = rescale + np.min(dataframe['Tnuc'])
    return translate

def return_model_plot(model):
    '''
    Keras.utils plot_model doesnt return figure object
    So save it to a file, read it back in so can then provide to summarise_model()
    '''
    dummy_file = 'tmp.png'
    plot_model(model, show_shapes=True, to_file=dummy_file)
    plt.rcParams['figure.figsize'] = [20, 50]
    im = plt.imread(dummy_file)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    os.remove(dummy_file)
    return fig

def ensemble_performance(members, X_TEST, Y_TEST, return_best=False, return_metrics = False,
                         return_ensem_metrics=False, preloaded=False):
    """
    Model
    :param members: members of the ensemble - i.e. list of model names
    :param X_TEST: test database
    :param Y_TEST: test labels
    :param return_best: whether to return best inidividual model of ensemble (MSE)
    :param return_metrics: whether to return metrics of each individual model
    :param return_ensem_metrics: whether to return metrics of ensemble
    :param preloaded: whether models in "members" are already loaded or not (faster if are)
    :return:
    """
    from keras.models import load_model
    from sklearn.metrics import r2_score, max_error
    from scipy.stats import spearmanr
    MAE_list=[]
    MSE_list=[]
    R2_list=[]
    spear_list=[]
    # ME_list=[]
    ypred_ensemble = np.full(Y_TEST.flatten().shape, 0.0)
    for model in members:
        if preloaded == False:
            model = load_model(model)
        ypred = model.predict(X_TEST)
        # sum ensemble predictions
        ypred_ensemble += ypred.flatten()

        # take individuals metrics
        MAE = np.mean(np.abs(Y_TEST - ypred.flatten()))
        MAE_list.append(MAE)
        MSE = np.mean((Y_TEST - ypred.flatten()) ** 2)
        MSE_list.append(MSE)
        R2 = r2_score(Y_TEST, ypred.flatten())
        R2_list.append(R2)
        spear = spearmanr(Y_TEST, ypred.flatten())
        spear_list.append(spear[0])
        # ME = max_error(Y_TEST, ypred.flatten())
        # ME_list.append(ME)

        # deal with multiple prediction memory leak
        import gc
        _ = gc.collect()

    metrics = np.array(list(zip(MAE_list, MSE_list, R2_list, spear_list)))
    print('\n#### Ensemble summary:\nIndividual models:\nMAE MSE R2 SP\n{}'.format(metrics))
    # average ensemble prediction
    ypred_ensemble = ypred_ensemble/float(len(members))
    MAE = np.mean( np.abs(Y_TEST - ypred_ensemble)
                   )
    MSE = np.mean( (Y_TEST - ypred_ensemble) ** 2
                   )
    R2 = r2_score(Y_TEST, ypred_ensemble)
    spear = spearmanr(Y_TEST, ypred_ensemble)[0]
    # ME = max_error(Y_TEST, ypred_ensemble)
    ensem_metrics = np.array([MAE, MSE, R2, spear])
    print('Ensemble model:\n{}'.format(ensem_metrics))

    if return_best:
        return members[np.argmin(MSE_list)]
    if return_metrics:
        return metrics
    if return_ensem_metrics:
        return ensem_metrics


def plot_Multiregression_vs_target(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):
    plt.rc('font', size=50)
    plt.rcParams['hatch.linewidth'] = 4.0
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 3.0
    plt.rcParams['ytick.major.width'] = 3.0
    plt.rcParams['xtick.minor.width'] = 3.0
    plt.rcParams['ytick.minor.width'] = 3.0
    plt.rcParams['xtick.major.size'] = 8.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['ytick.major.size'] = 8.0
    plt.rcParams['ytick.minor.size'] = 3.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [Y_TRAIN.shape[1]*5*2,
                                      Y_TRAIN.shape[1]*5*2]
    fig = plt.figure()

    plot_index = 0

    ### Train
    ylearn = MODEL.predict(X_TRAIN)
    for i in np.arange(Y_TRAIN.shape[1]):
        plot_index += 1
        ax = fig.add_subplot('{0}2{1}'.format(Y_TRAIN.shape[1], plot_index))
        ax.scatter(Y_TRAIN[:, i], ylearn[:, i], s=50, label='{}'.format(i))
        ax.set_xlabel('actual value')
        ax.set_ylabel('learnt value')
        ax.legend()
        ax.set_box_aspect(1)

        # y = x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, lw=4)

    ### Test
    ypred = MODEL.predict(X_TEST)
    for i in np.arange(Y_TEST.shape[1]):
        plot_index += 1
        ax = fig.add_subplot('{0}2{1}'.format(Y_TEST.shape[1], plot_index))
        ax.scatter(Y_TEST[:, i], ypred[:, i], s=50, label='{}'.format(i))
        ax.set_xlabel('actual value')
        ax.set_ylabel('predicted value')
        ax.legend()
        ax.set_box_aspect(1)

        # y = x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, lw=4)

    plt.tight_layout()

def get_colour_traj(frame, n_frames, my_cmap=cc.cm.bmw): #colours, colours_grain, grain):
    '''Get correct colour for frame from colour map
    Makes cmap equally spaced for n_frames discretisation
    INPUTS:
    frame = frame want single colour for
    n_frames = total no. frames to make entire cmaps span
    my_cmap = cmap to use'''
    my_cmap.set_under(color='black')
    # extract colours from cmap (0-1, with Nsegs). Equal spaced. Then use get_colour to get correct seg for temps (needed as they are not equally spaced)
    N_cmap_segs = n_frames
    colours = my_cmap(np.linspace(0,1,N_cmap_segs)) # parse the cmap into N segs
    colours_frame = np.arange(N_cmap_segs)
    for i,val in enumerate(colours_frame):
        if val == frame:
            correct_c = colours[i]
    return correct_c

def slice_center(image, cropx, cropy):
    '''
    :param image: original image
    :param cropx: how much in x to include around center
    :param cropy: how much in y to include around center
    :return: sliced image
    '''
    y, x = image.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return image[starty:starty + cropy, startx:startx + cropx]

def take_larger_slice(slicer: tuple, scale):
    """
    Take a larger slice - by factor "scale"
    :param slicer: slicer tuple object (i.e. im[slicer] -> new subset im)
    :param scale: factor by which to scale new slice
    :return: new slicer tuple object
    """
    # calc size of axis for rescale
    size_x = (slicer[0].stop - slicer[0].start)*scale
    size_y = (slicer[1].stop - slicer[1].start)*scale
    # extra points to shift slicer start and stops by
    extra_x = (size_x - (slicer[0].stop - slicer[0].start) ) / 2.0
    extra_y = (size_y - (slicer[1].stop - slicer[1].start) ) / 2.0
    # check float value is same as integer - i.e. scale can be done exactly
    if round(extra_x,12).is_integer() == False or round(extra_y,12).is_integer() == False:
        raise ValueError(
            'Extra padding to rescale requested = ({0}, {1}) which is not integer so scale cant be done exactly'.format(extra_x, extra_y))
    # make new slicers
    slicer_x = slice(int(slicer[0].start - extra_x),
                     int(slicer[0].stop + extra_x),
                     slicer[0].step)
    slicer_y = slice(int(slicer[1].start - extra_y),
                     int(slicer[1].stop + extra_y),
                     slicer[1].step)
    return (slicer_x, slicer_y)

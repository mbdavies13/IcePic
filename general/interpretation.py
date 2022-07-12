import colorcet as cc

def my_shap_plot(images, SHAP_VALUES, Y_ACTUAL, Y_PRED, graphname,
                 cmap_shap='bwr', shap_global_norm=False, v_shift=0.0, wc_global_norm=False):
    """
    Make shap image

    :param images: images calculated shap values for
    :param SHAP_VALUES: shap_values for images as calc'd by DeepExplainer
    :param Y_ACTUAL: list of image labels
    :param Y_PRED: list of model predictions
    :param shap_global_norm: shap cmap values are globally normalised or not in plot
    :param v_shift: shift limits of SHAP cmap by v_shift
    :param wc_global_norm: water contact layer cmap values are globally normalised or not in plot
    :return: side by side plots of image, and shap overlayed image - one system per pdf page
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig_list=[]
    for i in range(images.shape[0]):

        plt.rc('font', size=30)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
        fig.patch.set_facecolor('xkcd:mint green')

        if wc_global_norm == True:
            vmin=0.0
            vmax=np.max(np.abs(images.flatten()))
        else:
            vmin=np.min(np.abs(images[i]))
            vmax=np.max(np.abs(images[i]))

        p1 = ax1.imshow(images[i],
                        cmap=cc.cm.kbc,
                        interpolation='lanczos',
                        vmin=vmin,
                        vmax=vmax)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.axis('off')

        ## SHAP IMAGE
        # cbars must be symmetric - have center at 0
        if shap_global_norm == True:
            vmin=-1 * (np.max(np.abs(np.array(SHAP_VALUES).flatten())) + v_shift)
            vmax=np.max(np.abs(np.array(SHAP_VALUES).flatten())) + v_shift
        else:
            vmin = -1 * (np.max(np.abs(SHAP_VALUES[0][i])) + v_shift)
            vmax = np.max(np.abs(SHAP_VALUES[0][i])) + v_shift
        p2 = ax2.imshow(SHAP_VALUES[0][i],
                        cmap=cmap_shap,
                        interpolation='lanczos',
                        vmin = vmin,
                        vmax = vmax
                        # # logarithmic symmetric colormap
                        # norm=colors.SymLogNorm(vmin = vmin, vmax = vmax,
                        #                        linthresh=0.05, base=10)
        )
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.axis('off')

        plt.subplots_adjust(wspace=0.0)
        fig_list.append(fig)
        plt.close()

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(graphname)
    for fig in fig_list:
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')


'''
Script to make predictions of T on artificial images used for Fig. S5 of paper.
Process:
- Length scaling is achieved by zooming in and out but simple zomming changes respective atom sizes
- Here copy the original image, by locating its peaks and then place gaussians at those location by hand
- When place gaussians by hand can adjust their width linearly so they scale in the same way change change
... length scale of image

Final fig S5 taken by combining each of the 6 committee model predictions
    mean of the 6 = prediction
    stdev of the 6 = error bar

'''

############################################
import copy
import pandas as pd
from keras.models import load_model
import gc
from general.code import *
import glob

def detect_2dpeaks(image,threshold,neighborhood_size):
    import scipy.ndimage.filters as filters
    import scipy.ndimage as ndimage
    data = image
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2
        y.append(y_center)
    return maxima

def mean_height_peaks(image):
    '''
    Height of guassians to place
    -   take average peak heigh - but remove 10 % outside of image to
        avoid edge effects (seen this helps)
    See slice_center() in general.code to understand further
    :return: mean height of peaks - with 10% of edge trimmed
    '''
    # trim the image edges
    trimmed_im = slice_center(image,
                            # take 10 % off edge of picture
                            int(image.shape[0]*0.9),
                            int(image.shape[1]*0.9),
                            )
    # get peaks
    peaks = detect_2dpeaks(trimmed_im,threshold=0.03,neighborhood_size=20)
    # get mean height
    return np.mean(trimmed_im[peaks])


def change_image_gaussian_width(IMAGE, GRIDX, GRIDY, scale):
    '''
    Make a copy of an image with a scaled guassian width.
    Image must have single well defined peaks.

    :param IMAGE: original image to copy
    :param GRIDX: original gridx
    :param GRIDY: original gridy
    :param scale: contraction factor for gauss width
    :return: copied image with scaled gaussian width
    '''
    # 0) Detect peaks in the original image
    PEAKS = detect_2dpeaks(IMAGE, threshold=0.03, neighborhood_size=20)

    # 1) Make new image by placing guassians only at position of the peaks
    from general.xy_guass import gauss2d
    # gauss 2d adds one guassian at at time
    # --> loop over each peaks position and add
    IMAGE_COPY = np.full(GRIDX.shape, 0.0) # initiate new girdd
    for i in np.arange(len(GRIDX[PEAKS])):
        sys.stdout.write("\rPlacing guassian: " + str(i) + "/" + str(len(GRIDX[PEAKS]) - 1))
        sys.stdout.flush()
        z = gauss2d(# grid for gauss2d function
            xy=np.vstack([GRIDX.ravel(), GRIDY.ravel()]),
            # where to place guassians
            x0 = GRIDX[PEAKS][i],
            y0 = GRIDY[PEAKS][i],
            # gaussian properties
            amp=100.0, # leave height (change outside)
            a=125*scale,b=125*scale,c=125*scale,
            # resolution of grid
            dx=0.001
        )
        # reheight the gaussian so its same as original image peak
        # +0.05 to get closer to original
        z = z * ( (mean_height_peaks(IMAGE) + 0.05) / np.max(z) )
        # gauss2d takes 1D unravled grids - so make back into grid
        z.shape = GRIDX.shape
        # then add gauss to the grid
        IMAGE_COPY += z

    return IMAGE_COPY

############################################
IM_SIZE=50
## Read in dataframe
df = pd.read_csv('../dataframe.txt', sep=" ",header=0)
## Create normalised Tn column (do to datbase before stratify - needed as have ReLU hidden layers)
df['Tnuc_norm'] = normalise_Tn(df['Tnuc'])
print(df)

###### some handy variables #######
## showing xy plots of scaling
show_xy_plots = False # if dont show xy plots then the prediction plt
## scales to try
d_step = 0.02
list_scales = np.round(np.arange(0.8, 2.0 + d_step, d_step), 2)  # round to stop float point errors
##############################################
list_sysname = [
    # square
    'OH_Ic001_Cfull',
    # hexagon
    'LJ_fcc111_a2e5',
    # rhombus - 2X needed for highest scaling
    'OH_basal_full_0.05_2X',
    # rectangle
    'LJ_fcc110_a10e9',
]
list_pattern = ['sq', 'rect', 'rhomb', 'hex']

#### Setup models to make predictions
for STRAT_N in [1, 2, 3, 4, 5, 6]:

    print('##### STRAT:', STRAT_N, '\n')
    DIR = '../../CNN_Models/ProductionModel/STRAT{}'.format(STRAT_N)
    # Ensure below is correct path to the model names
    models = glob.glob('{0}/models/mod*'.format(DIR))
    print(np.array(models))
    ## load models now - ready to call later
    list_models = []
    for m in models:
        list_models.append(load_model(m))
    print('\n###### Models loaded in:\n', list_models)

    ## plot global settings ##
    plt.rc('font', size=30)
    plt.rcParams['axes.linewidth'] = 3.0
    # major/minor ticks
    plt.rcParams['xtick.major.width'] = 4.0
    plt.rcParams['ytick.major.width'] = 4.0
    plt.rcParams['xtick.minor.width'] = 2.0
    plt.rcParams['ytick.minor.width'] = 2.0
    plt.rcParams['xtick.major.size'] = 12.0
    plt.rcParams['ytick.major.size'] = 12.0
    plt.rcParams['xtick.minor.size'] = 6.0
    plt.rcParams['ytick.minor.size'] = 6.0
    # legend thickness
    plt.rcParams['legend.edgecolor'] = 'k'
    # figure size
    plt.rcParams['figure.figsize'] = [15, 10]
    if show_xy_plots:
        pass
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ######################
    i=0
    list_color = ['navy', 'xkcd:golden', 'skyblue', 'xkcd:orange']
    for sysname in list_sysname:
        i+=1
        print('\n### System:',sysname)
        path = '../../structures/{}/wat_zpred.txt'.format(sysname)
        # additional system 2X used for rhombus pattern
        if sysname == 'OH_basal_full_0.05_2X':
            path = '../../extra_system/OH_basal_full_0.05_2X/wat_zpred.txt'.format(sysname)
        # read in zpred
        ZPRED_OG, gridx, gridy = read_in_zpred(path, return_grid=True)
        # * KEY * normalise ZPRED_OG in same way do for cuts - so make correct height artificial image
        ZPRED_OG = ZPRED_OG / 0.4

        if show_xy_plots:
            image_show(ZPRED_OG, show_cmap=True)
            plt.show()

        # peaks
        peaks = detect_2dpeaks(ZPRED_OG,threshold=0.03,neighborhood_size=20)
        if show_xy_plots:
            plt.scatter(gridx,gridy,c=ZPRED_OG,marker='x', cmap=cc.cm.kbc)
            plt.scatter(gridx[peaks],gridy[peaks],c='r', marker='x')
            plt.show()

        ###### Do scaling
        list_ypred = []
        list_ypred_ensem = []
        for scale in list_scales:
            print('\nScaling by:', scale)
            # make artificial image with correct guassian width
            ZPRED_ART = change_image_gaussian_width(ZPRED_OG, gridx, gridy, 1.0/scale)
            if show_xy_plots:
                image_show(ZPRED_ART)
                plt.show()

            # cut to highest density - get slicer from original non artificial image location
            _, slicer_og = make_zpred_squarecut(ZPRED_OG, limit=200)
            ## ** by hand fix for hexagonal system - as 2.0 doesnt fit shift slicer og very slightly **
            if sysname == 'LJ_fcc111_a2e5':
                # shift slice in x (slicer_og[0]) so will fit at up to 2.0 zoom
                slicer_og = (slice(int(slicer_og[0].start + 50),
                                      int(slicer_og[0].stop + 50),
                                      slicer_og[0].step
                                      ),
                             slicer_og[1])
            ## ** by hand fix for rhombus system - as 2.0 doesnt fit shift slicer og very slightly **
            if sysname == 'OH_basal_full_0.05_2X':
                # shift slice in x (slicer_og[0]) so will fit at up to 2.0 zoom
                slicer_og = (slice(int(slicer_og[0].start + 150),
                                      int(slicer_og[0].stop + 150),
                                      slicer_og[0].step
                                      ),
                             slicer_og[1])

            # Then apply this slicer to artificial zpred - so that artificial slice matches original to best ability
            zpred_sq = ZPRED_ART[slicer_og]
            # original image
            zpred_og = resize(zpred_sq, (IM_SIZE, IM_SIZE))

            ######
            ## do scaled slice of artificial image
            # zooming in on image -  crop then rescale
            if scale < 1.0:
                if round((IM_SIZE * scale), 12).is_integer() == False:
                    raise ValueError(
                        'New image size requested {} is not integer so scale cant be done exactly'.format(IM_SIZE * scale))
                zscale = slice_center(zpred_og, int(IM_SIZE * scale), int(IM_SIZE * scale))
                # rescale to original size
                zscale = resize(zscale, (IM_SIZE, IM_SIZE))

            # zooming out of image - need larger cut - then rescale
            elif scale > 1.0:
                # take larger cut centered at same position as original image
                new_slicer = take_larger_slice(slicer_og, scale)
                zscale = ZPRED_ART[new_slicer]  # slice the full original zpred
                # rescale to original size
                zscale = resize(zscale, (IM_SIZE, IM_SIZE))

            elif scale == 1.0:
                zscale = copy.deepcopy(zpred_og)

            if show_xy_plots:
                image_show(zscale, show_cmap=True)
                plt.show()

            ## make predictions on image
            ypred_ensem = []
            m = 0
            for model in list_models:
                m += 1
                sys.stdout.write("\rModel: " + str(m) + "/" + str(len(list_models)))
                sys.stdout.flush()

                ypred_ensem.append(
                    model.predict(zscale.reshape(1, zscale.shape[0], zscale.shape[1], 1)).flatten()
                )

                # deal with multiple prediction memory leak
                # K.clear_session() # 1st solution
                _ = gc.collect()  # 2nd solution

            # store ensemble predictions
            list_ypred_ensem.append(ypred_ensem)
            # store average of ensemble
            list_ypred.append(np.sum(ypred_ensem) / len(list_models))

        list_ypred = convert_normT_2_unnormT(list_ypred, df)
        print('\nEnsemble predictions:', list_ypred)

        if show_xy_plots == False:
            ax.plot(list_scales, list_ypred, c=list_color[i - 1], lw=6, zorder=0,
                    label=list_pattern[i - 1])

    if show_xy_plots == False:
        # axis
        ax.legend(ncol=4, framealpha=0.25)
        ax.set_yticks(np.arange(200, 280, 10))
        ax.set_xticks(np.arange(0.2, 2.5, 0.1))
        ax.set_ylim(200, 272.5)
        ax.set_xlim((list_scales[0], list_scales[-1]))
        ax.set_ylabel('Tn')
        ax.set_xlabel('Scale')
        # ticks
        ax.tick_params(which='major', pad=15,
                       bottom=True, top=True,
                       left=True, right=True)
        ax.tick_params(which='minor',
                       bottom=True, top=True,
                       left=True, right=True)
        ax.tick_params(which="major", direction='in')
        ax.tick_params(which='minor', direction='in')
        ax.minorticks_on()

        plt.show()
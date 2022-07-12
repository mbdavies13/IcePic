from general.xy_guass import *
from general.code import image_show
from general.pz import *

def make_xy_water(sysname, TRAJ_SPLITS, path, show_plots=False, just_pz=False, return_grids=False,
                  t0=1, tf=-1, z_limit=None, flip=False, outFILE='wat_zpred.txt',
                  # args that are not needed if have default database structure
                  topoFILE=None, trjFILE=None,  name_water=None, boxarea=None, atom_selection=None, zpredFILE=None
                  ):
    """
    Function to make water contact layer xy images. Has two broad steps:
    1) perform p(z) analysis. Locate contact layer region as the trough after the largest peak.
        - also detect for doublet peaks via percentage change criterion
    2) make the xy image by placing gaussians on the a grid. Taking atoms from region identified in (1)

    :param sysname: name of the system - as find in system database
    :param show_plots: whether to show plots in addition to saving them
    :param path: path to the database
    :param just_pz: whether to just do the pz plots and not the xy images - so can check regions are correct
    :param return_grids: whether to return the gridx & gridy the zpred was made on
    :param t0, tf: first and final frame to take. t0=1 as 0th frame is topology file
    :param z_limit: if passed then will set a "roof" on how far in z histogram will be calculated (useful if two slab system)
    :param atom_selection: selection string for atoms to include in water contact layer - optional (name_water must be None)
    :param outFILE: name of zpred file output
    :return: pz_pics inside wat_pz_pics, wat_zpred.txt inside database, and zpred pic inside wat_zpred_pics
    """
    ### Read in the system and take z coordinates of the water atoms ###
    # topoFILE, trjFILE, name of water molecules and boxarea

    # if dont pass then try to extract from default file structure of database
    if topoFILE == None:
        topoFILE = '{0}/{1}/struct/struct2.pdb'.format(path, sysname)
    if trjFILE == None:
        trjFILE = '{0}/{1}/lammps/density2.xtc'.format(path, sysname)

    # if dont pass name_water then try to extract from file
    if name_water == None:
        if atom_selection == None: # if dont pass atom selection string - try via name_water arg
            name_water = np.genfromtxt(
                '{0}/{1}/struct/name_water.dat'.format(path, sysname), dtype='str')
            try:  # try to convert '1' to integer. If e.g. 'O' then fails and no need to convert
                name_water = int(name_water)
            except:
                pass

            # read in trajectory
            frames, waterpos_traj = trajandpos_xtc(topoFILE, trjFILE,
                                                   atom_selection='name {}'.format(name_water))
        # if want to pass atom selection string instead
        else:
            # read in trajectory
            frames, waterpos_traj = trajandpos_xtc(topoFILE, trjFILE,
                                                   atom_selection='{}'.format(atom_selection))

    # if dont pass boxarea then try to extract from file
    if boxarea == None:
        boxarea, _, _, _ = np.loadtxt(
            '{0}/{1}/struct/area_struct2.txt'.format(path, sysname))

    # flip option
    if flip:
        waterpos_traj = np.array(waterpos_traj)
        waterpos_traj[:, :, 2] = waterpos_traj[:, :, 2] * -1.0 # flip in z

    # extract z coordinates for water
    z = np.array(waterpos_traj)[:, :, 2]


    #################################################################################
    ###################### Perform pz_histogram analysis ############################

    ## calculate pz ##
    if z_limit == None:
        # normal way
        bincenters, pz_hist, pz_peaks, pz_troughs = calc_pz_hist(z.flatten(),
                                                                 bin_width=0.04,
                                                                 norm=True,
                                                                 frames=frames,
                                                                 boxarea=boxarea)
        # if dont have a trough after largest peak try again with lower prominence for troughs
        if pz_peaks[np.argmax(pz_hist[pz_peaks])] >= pz_troughs[-1]:
            i=5
            while pz_peaks[np.argmax(pz_hist[pz_peaks])] >= pz_troughs[-1] and i > 1:
                i -= 1
                bincenters, pz_hist, pz_peaks, pz_troughs = calc_pz_hist(z.flatten(),
                                                                         bin_width=0.04,
                                                                         norm=True,
                                                                         frames=frames,
                                                                         boxarea=boxarea,
                                                                         trough_prom=i)

    else:  # cut all values higher than passed z_limit
        bincenters, pz_hist, pz_peaks, pz_troughs = calc_pz_hist(z.flatten()[z.flatten() <= z_limit],
                                                                 bin_width=0.04,
                                                                 norm=True,
                                                                 frames=frames,
                                                                 boxarea=boxarea)



    ## make list of trough locations - and ensure troughs surround required peaks ##
    # make first trough = the bin before the first non-zero part of histogram
    pz_troughs = np.append(np.min(np.nonzero(pz_hist)) - 1, pz_troughs)
    # trough locations
    troughsloc = bincenters[pz_troughs]
    maxpeaksloc = bincenters[pz_peaks][np.argmax(pz_hist[pz_peaks])]

    ## detect doublet ##
    doublet: bool = False  # default is False
    if len(pz_peaks) > 1:
        # if largest peak is also last peak - then not possible to take a doublet
        if pz_peaks[np.argmax(pz_hist[pz_peaks])] == pz_peaks[-1]:
            doublet = False
        # if peak after largest within threshold
        elif (pz_hist[pz_peaks][np.argmax(pz_hist[pz_peaks])] - pz_hist[pz_peaks][np.argmax(pz_hist[pz_peaks]) + 1]) / \
                pz_hist[pz_peaks][np.argmax(pz_hist[pz_peaks])] * 100 < 5.0:
            doublet = True
        # if trough after largest peak is still high density
        elif pz_hist[pz_troughs[np.argmax(pz_hist[pz_peaks]) + 1]] > 40:
            doublet = True


    ## create z_regions and plot exact region where z_calc indexs will take
    # set z_regions
    z_regions = []
    for i in np.arange(len(troughsloc)-1):
        i = i + 1
        if i < (len(troughsloc)):
            z_regions.append(list(np.take(troughsloc, [i-1, i])))
    # find region to take
    for i, val in enumerate(z_regions):
        if val[0] < maxpeaksloc < val[1]:
            mainpeak_index = i
    if type(mainpeak_index) != int:
        raise ValueError('Location of largest peak in p(z) of water not achieved. Must be issue with peak/trough detection in p(z)')
    if doublet == True:
        print(
            "\n\n DOUBLET \n Doublet deteced - so taking one more peak than usual script protocol does.",
            "One more peak --> one more trough. Check via z graphs if appropriate\n")
        print("z regions are:", z_regions)
        region = np.array([z_regions[0][0], z_regions[mainpeak_index][-1]])
        print("Region would have been:", region)
        mainpeak_index = mainpeak_index + 1  # now increase to next index

        # deal with possibility of not having trough after new largest peak
        if pz_peaks[mainpeak_index] >= pz_troughs[-1]:
            print('\nTaking doublet not possible currently due to no trough being after new final peak..')
            print('Adjusting trough prominence and recalculating regions')
            i=5
            while pz_peaks[mainpeak_index] >= pz_troughs[-1] and i > 1:
                i -= 1
                bincenters, pz_hist, pz_peaks, pz_troughs = calc_pz_hist(z.flatten(),
                                                                         bin_width=0.04,
                                                                         norm=True,
                                                                         frames=frames,
                                                                         boxarea=boxarea,
                                                                         trough_prom=i)
                print(i)
            # re do z regions once while loop finished
            pz_troughs = np.append(np.min(np.nonzero(pz_hist)) - 1, pz_troughs)
            troughsloc = bincenters[pz_troughs]
            z_regions = []
            for i in np.arange(len(troughsloc) - 1):
                i = i + 1
                if i < (len(troughsloc)):
                    z_regions.append(list(np.take(troughsloc, [i - 1, i])))
            print("z regions are now:", z_regions)

        # continue..
        region = np.array([z_regions[0][0], z_regions[mainpeak_index][-1]])
        print("Now is:", region)
    # For water: take from start - up to trough after the largest peak.
    region = np.array([z_regions[0][0], z_regions[mainpeak_index][-1]])
    zbot = region[0]
    ztop = region[-1]
    print("## Extracting indexes for water:")
    print("Plotting xy histogram for z region (nm):", region)

    ## plot pz histogram ##
    # full histogram - option to show
    plot_pz_hist(z, bincenters, pz_hist, pz_peaks, pz_troughs,
                           show_bool=show_plots)
    plt.close('all')
    # area of interest
    fig, ax = plot_pz_hist(z, bincenters, pz_hist, pz_peaks, pz_troughs,
                           show_bool=False)
    ax.set_xlim(right=(np.min(z)+3.5) if (np.min(z)+3.5) > region[-1] else (region[-1]+1.0))
    # annotate figure - doublet
    ax.text(0.0, 1.0,
        'System: {0}\n Doublet: {1}'.format(sysname, doublet),
        fontsize=20, ha='left', va='top', transform=fig.transFigure
    )
    # plot potential z regions
    for i, val in enumerate(troughsloc):
        ax.axvline(val, c='navy', linestyle='dashed', lw=3)
    # plot region where will make xy image
    ax.plot(region, [np.mean(pz_hist), np.mean(pz_hist)], c='navy', lw=3)

    if show_plots:
        plt.show()

    # if just doing pz then stop here
    if just_pz == True:
        return

    #################################################################################
    ########################   Make water xy image   ################################
    print('\n### Calculating xy image')
    #### Extracting the molecules for the desired region ####
    print('Finding atoms in region {}...'.format(region))
    z_list=[]
    for i in np.arange(len(waterpos_traj)):
        z_index=[]
        for j,val in enumerate(waterpos_traj[i][:,2]):
            if zbot <= val < ztop:
                z_index.append(j)
        tmp=np.array(z_index)
        z_list.append(tmp)

    # get edges for making grid
    hist, xedges, yedges = histogram_calc(waterpos_traj, z_list=z_list, frames=frames, plot=False,
                                          zbot=zbot,
                                          ztop=ztop)

    # get z regions for x and y positions
    xpos_zregion, ypos_zregion =  STRUC_zregionextract_xypos_water(z_list,waterpos_traj)
    # cut to deisred frames. Defualt is to take all (i.e. [0:-1])
    xpos_zregion = xpos_zregion[t0:tf]
    ypos_zregion = ypos_zregion[t0:tf]

    # make gauss image
    zpred, gridx, gridy, conv_factor = gauss_exact(waterpos_traj,xpos_zregion=xpos_zregion,ypos_zregion=ypos_zregion,
                                                   xedges=xedges, yedges=yedges,
                                                  water=True,
                                                  plot2d=False, plot2d_where=False, plot3d=False, plot_conv=False,
                                                  split_trajectory=True,traj_splits=TRAJ_SPLITS)

    # plot and save image
    image_show(zpred)
    # plt.savefig('PATH/{}.png'.format(sysname))
    if show_plots:
        plt.show()

    # save the zpred
    # if dont provide name then assume default structure
    if zpredFILE == None:
        f = open('{0}/{1}/{2}'.format(path, sysname, outFILE), "wb")
        pickle.dump([zpred, gridx, gridy, conv_factor], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        f = open('{0}'.format(zpredFILE), "wb")
        pickle.dump([zpred, gridx, gridy, conv_factor], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    if return_grids == True:
        return gridx, gridy, zpred
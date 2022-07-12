import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import sys
import itertools 
from scipy.ndimage import rotate
import math
from joblib import Parallel, delayed
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import MDAnalysis
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial import distance

####################################################################################################
def picklereader(pickleFILE):                                                              
    """function to read pickle file and assign variables accordingly"""                                
    f=open(os.path.abspath(pickleFILE))                                                    
    bulkloc, peaksloc, troughsloc, peaks, troughs, maxpeaksloc, sub_troughs, sub_peaks, subpeaksheight, sub_zbincenters = pickle.load(f)                                              
    f.close()                                                                              
    return bulkloc, peaksloc, troughsloc, peaks, troughs, maxpeaksloc, sub_troughs, sub_peaks, subpeaksheight, sub_zbincenters                                                        

def picklereader_ice(pickleFILE):
    f=open(os.path.abspath(pickleFILE))
    sub_troughsloc = pickle.load(f)
    f.close()
    return np.concatenate(sub_troughsloc) #np.conc to stop it being list(array([])) struc.

def read_zcuticeface(pickleZCUT):
    f=open(os.path.abspath(pickleZCUT))
    zcut, gridx_cut, gridy_cut = pickle.load(f)
    f.close()
    return zcut, gridx_cut, gridy_cut

                                                                                           
####trajectory and positions of atoms
def trajandpos_xtc(topoFILE, trjFILE, atom_selection):
    """
    Function to read in topology and trajectory file like vmd.
    """
    print("Reading trajectory.. \n traj and topo files:\n", trjFILE, "\n", topoFILE)
    start_time = time.time()
    u = MDAnalysis.Universe(topoFILE, trjFILE)
    natom = len(u.select_atoms('{}'.format(atom_selection), updating=True))
    atoms = u.select_atoms('{}'.format(atom_selection), updating=True)
    if natom == 0:
        raise ValueError('Zero atoms found. Provide correct selection str.')
    pos_traj=[]
    for ts in u.trajectory:
        pos_traj.append(atoms.positions * 0.1)
    frames = len(pos_traj)
    print("Program took", time.time() - start_time, "to read trajectory and calculate position of atoms")
    print("Found", frames, "frames and", natom, "atoms")
    return frames, pos_traj

def trajandpos_ase(topoFILE, trjFILE, atom_selection):
    '''
    Function to read in topology and trajectory file like vmd and give ASE trajecotry

    :param topoFILE: topology file
    :param trajFILE: trajecotry file
    :return: frames, position trajecotry
    '''
    import ase
    import MDAnalysis
    import time
    import string
    start_time=time.time()
    print("Reading trajectory.. \n traj and topo files:\n", trjFILE, "\n", topoFILE)
    atoms_list = []
    u = MDAnalysis.Universe(topoFILE, trjFILE)
    atoms = u.select_atoms('{}'.format(atom_selection), updating=True)
    for ts in u.trajectory:
        positions = atoms.positions
        # try to give symbols - but ase needs letters so may fail
        try:
            names = atoms.names
            atoms_list.append(ase.Atoms(symbols=names,
                                        positions=positions,
                                        cell=ts.dimensions,
                                        pbc=True))
        except: # just dont give symbols
            atoms_list.append(ase.Atoms(
                                        positions=positions,
                                        cell=ts.dimensions,
                                        pbc=True))
    frames = len(u.trajectory)
    natom = len(atoms)
    print('Trajectory has', frames, 'frames with', natom, 'atoms')
    print('Took', time.time()-start_time,'to read traj')
    return frames, atoms_list


def pos_from_topo(topoFILE, atom_selection):
    """Function to read in topology and trajectory file like vmd.
    KEY: Takes positions of waters and substrate based on idea that they never change
    order in file. i.e. atom 0 always first line of file. AND that water comes first."""
    print("Reading topo file:", topoFILE)
    u = MDAnalysis.Universe(topoFILE)
    natom = len(u.select_atoms('{}'.format(atom_selection), updating=True))
    if natom == 0:
        raise ValueError('Zero atoms found. Provide correct selection string.')
    atoms = u.select_atoms('{}'.format(atom_selection), updating=True)
    pos = atoms.positions*0.1
    print("Found", natom, "atoms")
    return 1, pos

def hexpos_xtc(topoFILE):
    u = MDAnalysis.Universe(topoFILE)
    for ts in u.trajectory:
        continue
    frames = 1 #only take one for substrate
    zsub = ts.positions[:]*0.1 #just take for one frame
    nsub = len(zsub)
    print("Number of substrate", nsub, "\nNumber of frames taken for substrate", frames)

    return frames, zsub

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def pairwise(iterable): 
    """function to iterate over list of positions of troughs
    s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def sub_significantpeaks(sub_peaks_i,sub_peaksheight_i,sub_troughs_i,threshold=50):        
    """Function to find insiginifcant peaks in the substrate. Then removes the             
    appropriate troughs so don't make seperate images of xy just for them"""               
    if type(sub_troughs_i) == str and sub_troughs_i == 'full':                             
        print("Substrate has no peaks and just taking full extent in z")
        h='full'                                                                           
        return h                                                                           
    else:                                                                                  
        print("#######Substrate has peaks - Sorting peaks of substrate#########")
        print("Insignificant peaks were removed. Check done correctly: \nInitial peaks:",
              "peaks height, and troughs read in:", sub_peaks_i,sub_peaksheight_i,sub_troughs_i)
        if len(sub_troughs_i) != (len(sub_peaks_i)+1):                                     
            raise Exception("Error in z histogram script that is passed. There is not one more trough than peaks. That is expected for this")
        else:                                                                              
            idelete=[]                                                                     
            for i,val in enumerate(sub_peaksheight_i):                                     
                if val <= threshold:                                                       
                    idelete.append(i)                                                      
        h=sub_troughs_i                                                                    
        for it0 in np.arange(len(idelete)):                                                
            h = np.append(h,sub_peaks_i[idelete][it0])                                     
        h = np.sort(h)                                                                     
        for it0 in np.arange(len(idelete)):                                                
            print("insignificant peak:", sub_peaks_i[idelete[it0]],", trough right of insig peak:", sub_troughs_i[idelete[it0]+1],", trough left of insig peak:", sub_troughs_i[idelete[it0]])
            h=np.delete(h,np.where(h==sub_troughs_i[idelete[it0]])[0][0])                  
            h=np.delete(h,np.where(h==sub_peaks_i[idelete[it0]])[0][0])                    
        ##since chuck away the trough to the left of the insignificant peak. Need to be sure 
        ##..that take the first trough                                             
        if h[0] != 0:                                                                      
            print("frist sub trough moved to start of sub")
            h[0]=0                                                                         
        if h[0] !=0 or h[-1] != (len(sub_zbincenters)-1):                                  
            raise Exception("start or end of substrate is not included! Investigate")
        else:                                                                              
            print("troughs taken:", h)
            print("bincenters that result from this:", sub_zbincenters[h])
        ###testing substrate peaks and troughs
        test = np.sort(np.append(sub_peaks_i,sub_troughs_i))                               
        print("Checking order of peaks and troughs when put took together in one list and sort in height in z. Order should be trough,peak,trough,peak..")
        r=[]                                                                               
        for i in np.arange(0,len(test),2):                                                 
            if len(np.where(sub_troughs_i==test[i])[0]) == 0:
                r.append(False)
                raise Exception("peak and trough order is not correct. Test returned False. MUST FIX!!!")
            else:                                                                          
                r.append(True)                                                             
        if np.all(r) == True:                                                              
            print("test was successfull!")
                                                                                           
        print("#########End of peak sorting###########")
        return sub_zbincenters[h] 
                                                                                           
def zcalc_indexes(pos_traj,troughsloc=[],sub_troughsloc=[],setregion=False,region=[],water=False,substrate=False,subregion=[],start_index=0,doublet=False):
    #start_index only used if want to look at ice faces and would like to look at not just..
    #..[top,1st], [top,2nd].. e.g. start_index=1 then can look at [1st,2nd], [1st,3rd] etc.
    """Function that calculates the zindexes for main peak of water over the different     
    frames. Conditional to set own region.                                                   
    pos_traj = waterpos_traj if for water, or subpos if for substrate"""                   
                      
    #######First three outer if conditionals - decide which region you look at##########   
    #1) take from position of water troughs - (water)                                      
    #2) set your own region to look at - (water or sub)                                    
    #3) take from position of substrate troughs - (substrate)                              
    if setregion == False and water == True:                                               
        if substrate == True:#check not used wrong keywords                                
            raise Exception("Can't run function for water and substrate simulataneously. Check keywords")
        else:                                                                              
            if troughsloc==[]:                                                             
                raise Exception("Did not pass water trough positions to function")
            # set z regions from troughsloc
            z_regions = []
            for i in np.arange(len(troughsloc) - 1):
                i = i + 1
                if i < (len(troughsloc)):
                    z_regions.append(list(np.take(troughsloc, [i - 1, i])))
            # find region to take
            for i, val in enumerate(z_regions):
                if val[0] < maxpeaksloc < val[1]:
                    mainpeak_index = i
            if doublet == True:
                print("\n\n DOUBLET \n Told script this is a doublet case - so taking one more peak than usual script protocol does. One more peak --> one more trough. Check via z graphs if appropriate\n")
                print("All the regions are:", z_regions)
                region=np.array([z_regions[0][0],z_regions[mainpeak_index][-1]])
                print("Region would have been:", region)
                mainpeak_index=mainpeak_index+1 #now increase to next index
                region=np.array([z_regions[0][0],z_regions[mainpeak_index][-1]])
                print("Now is:", region)
            #For water: take from start - up to trough after the largest peak.
            region=np.array([z_regions[0][0],z_regions[mainpeak_index][-1]])
            zbot=region[0]
            ztop=region[-1]
            print("## Extracting indexes for water:")
            print("Plotting xy histogram for z region (nm):", region)
    elif setregion == True:                                                                
        z_regions='Null'                                                                   
        if region == []:                                                                   
            print("Need to pass region arguement to variable")
        else:                                                                              
            print("####Water or substrate - taking region set by hand instead of from z histogram")
            print("plotting xy histogram for z region (nm):", region)
            zbot=region[0]                                                                 
            ztop=region[-1]                                                                
    elif setregion == False and substrate == True:                                         
    ##loop takes from top of substrate, to penultimate trough. Then top to 2nd last trough etc.
        if water == True: #check not used wrong keywords                                   
            raise Exception("Can't run function for water and substrate simulataneously. Check keywords")
        else:                                                                              
            if sub_troughsloc==[]:                                                         
                raise Exception("Did not pass substrate trough positions to function")
            elif sub_troughsloc == 'Full':                                                 
                zbot = np.min(pos_traj[:,2])                                               
                ztop = np.max(pos_traj[:,2])                                               
                z_regions = [zbot, ztop]                                                   
            else:                                                                          
                print("## Extracting indexes for substrate:\nTroughs passed to substrate:", sub_troughsloc)
                z_regions=[]                                                               
                for i in np.arange(len(sub_troughsloc)-1):
                    i=i+1+start_index
                    if i < (len(sub_troughsloc)):
                        z_regions.append(list(np.take(np.flip(sub_troughsloc),[start_index,i])))
                print("Regions can look at of substrate:", z_regions)

    ###################Extracting the molecules for the desired region##################   
    ##water: z_list = list of arrays, for each frame the indexes of atoms that             
    ##..lie in the set region of zbot and ztop                                             
    ##sub: only one frame. z_list = the array for that frame. i.e. has one less dim than   
    ##..the water array. water z_list[0] equiv. to sub z_list                              
    z_list=[] #only for one value of zbot and ztop                                         
    if water == True:       
        if substrate == True: #check not used wrong keywords                                
            raise Exception("Can't run function for water and substrate simulataneously. Check keywords")
        elif len(np.array(pos_traj).shape) != 3:                                           
            raise Exception("Water pos trajectory not of correct shape. Likely passed the sub positions instead of the water positions")
        else:                                                                              
            for i in np.arange(frames):                                                    
                z_index=[]                                                                 
                for j,val in enumerate(pos_traj[i][:,2]):                                  
                    if zbot <= val < ztop:                                                 
                        z_index.append(j)                                                  
                tmp=np.array(z_index)                                                      
                z_list.append(tmp)                                                         
                tmp=[]                                                                     
    if substrate==True: #only need to carry out for one frame as substrate is frozen
        if len(pos_traj.shape) != 2:                                                       
            raise Exception("More than one frame of positions passed to substrate. Likely passed water trajectory by mistake")
        elif sub_troughsloc=='Full':                                                       
            z_list = list( np.arange(len(pos_traj)) )
        elif setregion == True:                                                            
            z_regions='Null'                                                               
            for j,val in enumerate(pos_traj[:,2]):                                         
                if zbot <= val < ztop:                                                     
                    z_list.append(j)                                                       
        else:                                                                              
            if subregion == list:                                                          
                raise Exception("Did not tell function which region you want to look at based of the positions of the troughs")
            else:                                                                          
                print("Region of substrate extracting atoms for:", z_regions[subregion])
                z_list=[]                                                                  
                zbot = z_regions[subregion][-1]                                            
                ztop = z_regions[subregion][0]                                             
                for j,val in enumerate(pos_traj[:,2]):                                     
                    if zbot <= val < ztop:                                                 
                        z_list.append(j)                                                   
                                                                                           
    return z_regions, region, zbot, ztop, z_list                                           

def magnitude(x):
    return int(math.floor(math.log10(x)))

def OLD_zregionextract_xypos_water(z_list,waterpos_traj):
    """function to calculate xy positions for a zregion.
    The names of the arguements are set to the defualt names for the water. But this
    function can be used for any other z_list or waterpos_traj type object.
    Not strucuted into list of length frames. If need that then use function below"""
    pos_zregion=[] #a list of all the positions for the desired indexes for the trajectory.
    for it0, val0 in enumerate(z_list):
        for it1, val in enumerate(z_list[it0]):
            pos_zregion.append(waterpos_traj[it0][val])
    xpos_zregion=[]
    ypos_zregion=[]
    for i in np.arange(len(pos_zregion)):
        xpos_zregion.append(pos_zregion[i][0])
        ypos_zregion.append(pos_zregion[i][1])
    print("number of samples in x and y have order of magnitude:", magnitude(len(xpos_zregion)),
          magnitude(len(ypos_zregion)), "is this reasonable? If not, likely made mistake with zcalc_index function")
    return xpos_zregion, ypos_zregion

def STRUC_zregionextract_xypos_water(z_list,waterpos_traj):
    """function to calculate xy positions for a zregion.
    Returns list of xpos and ypos in structured array list. len(list) = no. frames"""
    pos_zregion=[] #a list of all the positions for the desired indexes for the trajectory.
    #ordered into frames etc so can parrallerise the gaussian placing upon grid    
    for it0, val0 in enumerate(z_list):
        pos_zlist=[]
        for it1, val in enumerate(z_list[it0]):
            pos_zlist.append(waterpos_traj[it0][val])
        tmp=np.array(pos_zlist)
        pos_zregion.append(tmp)
        tmp=[]
    xpos_zregion=[]
    ypos_zregion=[]
    for it0, val0 in enumerate(pos_zregion):
        xpos_zlist=[]
        ypos_zlist=[]
        for it1, val in enumerate(pos_zregion[it0]):
            xpos_zlist.append(val[0])
            ypos_zlist.append(val[1])
        tmp=np.array(xpos_zlist)
        xpos_zregion.append(tmp)
        tmp=np.array(ypos_zlist)
        ypos_zregion.append(tmp)
        tmp=[]
    #magnitude approximated that each frame has same order of mag number of atoms in
    #..in region. i.e. len(xpos_zregion[0]) is representative of all frames. - fine as this is just sanity check to help user debug if making an error
    print("number of samples in x and y have order of magnitude:", magnitude(len(xpos_zregion)*len(xpos_zregion[0])),
          magnitude(len(ypos_zregion)*len(ypos_zregion[0])), "is this reasonable? If not, likely made mistake with zcalc_index function")
    return xpos_zregion, ypos_zregion

def extract_xypos_sub(subpos):
    """simple function to use to take all of substrate positions"""
    xpos_zregion = subpos[:,0]
    ypos_zregion = subpos[:,1]
    zpos_sub = subpos[:,2]
    print("number of samples in x and y have order of magnitude:", magnitude(len(xpos_zregion)),
          magnitude(len(ypos_zregion)), "is this reasonable? If not, likely made mistake with zcalc_index function")
    return xpos_zregion, ypos_zregion
    
def zregionextract_xypos_sub(z_list,subpos):                                                
    pos_zregion=[]                                                                          
    for i, val in enumerate(z_list):                                                        
        pos_zregion.append(subpos[val])                                                     
    xpos_zregion=[]                                                                         
    ypos_zregion=[]                                                                         
    for i in np.arange(len(pos_zregion)):                                                  
        xpos_zregion.append(pos_zregion[i][0])                                             
        ypos_zregion.append(pos_zregion[i][1])
    print("number of samples in x and y have order of magnitude:", magnitude(len(xpos_zregion)),  magnitude(len(ypos_zregion)), "is this reasonable? If not, likely made mistake with zcalc_index function")
    return xpos_zregion, ypos_zregion 

def histogram_calc(pos_traj, z_list, frames, zbot, ztop, water=True,sub='none',subpos=[],xwidth=0.1,ywidth=0.1, plot=False):
    """function to calculate histogram for system. Default setup to be looking at water with full simulation frames.
    Things that must be set outisde this function:                                  
    subpos, waterpos_traj,troughsloc"""
    if water == True:
        if sub != 'none':
            raise Exception("*********ERROR*********: water=True and sub != 'none'. Can't calculate histogram for both water and substrate at the same time. Correct the arguements passed to function")
        else:
            xpos_zregion, ypos_zregion = OLD_zregionextract_xypos_water(z_list,pos_traj)
    elif sub == 'Full': #taking all of substrate                                          
        frames = 1
        xpos_zregion, ypos_zregion = extract_xypos_sub(pos_traj)
    elif sub == 'region': #taking region of substrate                                        
        frames = 1
        xpos_zregion, ypos_zregion = zregionextract_xypos_sub(z_list,pos_traj)
    xdim=np.amax(xpos_zregion)-np.amin(xpos_zregion)
    print("extent of x taken=", xdim)
    ydim=np.amax(ypos_zregion)-np.amin(ypos_zregion)
    print("extent of y taken=", ydim)
    zdim=ztop-zbot
    print("extent of z taken", zdim)

    binsx = np.arange(np.amin(xpos_zregion),np.amax(xpos_zregion),xwidth)
    binsy = np.arange(np.amin(ypos_zregion),np.amax(ypos_zregion),ywidth)
    hist, xedges, yedges, im = plt.hist2d(xpos_zregion,ypos_zregion,bins=[binsx,binsy], cmap='Blues')
    plt.close()
    if plot == True:
        plt.close()
        hist, xedges, yedges, im = plt.hist2d(xpos_zregion,ypos_zregion,bins=[binsx,binsy],cmap='Blues', vmin=np.amin(hist/(zdim*frames*xwidth*ywidth)),vmax=np.amax(hist/(zdim*frames*xwidth*ywidth)))
        im.set_array(hist/(zdim*frames*xwidth*ywidth))
        plt.xlabel('y(nm)')
        plt.ylabel('x(nm)')
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 55
        cbar.ax.set_ylabel(r"$\rho (nm^{-3})$", rotation=270)
        plt.show()
    return hist, xedges, yedges

def extract_data(histogram,xbincenters,ybincenters): #, params):                            
    """use to extract bins from histogram data. Can comment or uncomment out the "zobs"     
    ..currently commented out as dont use this func to place gaussians anymore"""           
    xgauss=[]                                                                               
    ygauss=[]                                                                               
    for it0, val in enumerate(histogram):                                                   
        #it0 is the xbin                                                                    
        for it1,val in enumerate(histogram[it0]):                                           
            #it1 is the ybin                                                                
            if val != 0:                                                                    
                xgauss.append(xbincenters[it0])                                             
                ygauss.append(ybincenters[it1])                                             
    xy = np.array([xgauss,ygauss],dtype='float64')                                          
    #zobs = gauss2d(xy, *params)                                                            
    return xy

def test_gridy(gridy,res_width):                                                            
    """function to check made correct grid in y"""                                          
    test = np.around(np.diff(gridy[0]),10)                                                  
    if np.all(test == np.around(res_width,10)):                                             
        return True                                                                         
    else:                                                                                   
        return False                                                                        
def test_gridx(gridx,res_width):                                                            
    """function to check made correct grid in y"""                                          
    test = np.around(np.diff(gridx[:,0]),10)                                                
    if np.all(test == np.around(res_width,10)):                                             
        return True                                                                         
    else:                                                                                   
        return False                                                                        
                                                                                            
def guess_grid(xedges,yedges,res_factor,width,npad=1):                                             
    """This function makes initial guess at the grid. This is commonly wrong hence "guess" 
    Need the set of conditionals in make_grid function below to test it and make it         
    correct."""                                                                   
    #npad is number of bins to pad grid by          
    #instead of bincenters take the edges from the histogram - ensures that grid is large enough
    X = xedges
    Y = yedges
    #pad grid to have more bins at min and max (full bin hence +/- width)
    X=np.append(np.insert(X,0,X[0]-(npad*width)),X[-1]+(npad*width)) 
    Y=np.append(np.insert(Y,0,Y[0]-(npad*width)),Y[-1]+(npad*width))
    res_width = width*res_factor                                                            
    print("width of boxes in x and y for original data: ", width, "(nm)")
    print("width of boxes in x and y for the fine mesh, and shape: ", res_width, "(nm)")
                                                                                            
    ##first guess at grids
    #n's = number of boxes - caculate via np.arange that gives desired grid points basis    
    #np.mgrid uses a np.linspace type method where needs number of bins not width           
    nx = len(np.arange(X[0],X[-1],res_width)) #should end with X[-1] -- checked             
    ny = len(np.arange(Y[0], Y[-1],res_width))                                              
    gridx, gridy = np.mgrid[X[0]:X[-1]:complex(nx), Y[0]:Y[-1]:complex(ny)]                 
    return gridx,gridy,res_width,X,Y,nx, ny                                

def make_grid(xedges,yedges,res_factor=0.1,width=0.1,npad=1):                                      
    """This calls gues_grid function first. Then has list of conditionals to test the       
    grid that is guessed, via the test functions defined above. Then changes the grid       
    appropriately"""                                                                        
    gridx, gridy, res_width, X, Y, nx, ny = guess_grid(xedges,yedges,res_factor,width,npad) #make initial guess                                                            
    #######ensuring that gridx, gridy represents the original bins with finer mesh and consistent binwidth 
    if test_gridy(gridy,res_width) and test_gridx(gridx,res_width):                         
        print("gridx and gridy correct gaps from first initial code")
    elif test_gridy(gridy,res_width):                                                       
        nx = len(np.arange(X[0],X[-1],res_width)) + 1                                       
        gridx, gridy = np.mgrid[X[0]:X[-1]:complex(nx), Y[0]:Y[-1]:complex(ny)]             
    elif test_gridx(gridx,res_width):                                                       
        ny = len(np.arange(Y[0], Y[-1],res_width)) + 1                                      
        gridx, gridy = np.mgrid[X[0]:X[-1]:complex(nx), Y[0]:Y[-1]:complex(ny)]             
    elif not test_gridy(gridy,res_width) and not test_gridx(gridx,res_width):               
        nx = len(np.arange(X[0],X[-1],res_width)) + 1                                       
        ny = len(np.arange(Y[0], Y[-1],res_width)) + 1                                      
        gridx, gridy = np.mgrid[X[0]:X[-1]:complex(nx), Y[0]:Y[-1]:complex(ny)]             
    else:                                                                                   
        raise Exception("*****ERROR*****:investigate gridx and gridy - could not verify they are correct or correct them if incorrect")
    if test_gridy(gridy,res_width) and test_gridx(gridx,res_width):                         
        print("gridx and gridy had incorrect gaps but checked and fixed them")
    else:
        raise Exception("*****ERROR*****: gridx and gridy where incorrect gaps - checked them and attempted fix - this has not worked. Investigate")
    print("width of boxes in x and y for the fine mesh, and shape of gridx & gridy: ", res_width, ";", gridx.shape, gridy.shape)
    return gridx, gridy, X, Y, res_width                                        
                                                                                            
def gauss2d(xy, amp, x0, y0, a, b, c, dx):                                                 
    """Normalised 2d gaussian. Integral equals 1. dx is key parameter for integration      
    as must represent actual seperation of point for the gaussian plotted in the grid.     
    res_width is passed from make_grid function"""                                         
    x, y = xy                                                                              
    inner = a * (x - x0)**2                                                                
    inner += 2 * b * (x - x0)**2 * (y - y0)**2                                             
    inner += c * (y - y0)**2                                                               
    return (amp * np.exp(-inner))/(np.trapz(amp * np.exp(-inner),dx=dx))                   
                                                                                           
def gauss_frame_RAM(i,xyi,amp,a,b,c,res_width,gridx, xpos_zregion, ypos_zregion):
    """
    Function that is passed to parallelrisation.
    Called to parallerise process of adding gaussians to each frame.
    """
    zadd_list=[]
    zadd_oneframe=[]
    for it1, val1 in enumerate(xpos_zregion[i]):
        z = gauss2d(xyi,amp,xpos_zregion[i][it1],ypos_zregion[i][it1],a,b,c,dx=res_width)
        z.shape = gridx.shape
        if len(zadd_oneframe)==0:
            zadd_oneframe=z
        else:
            zadd_oneframe=zadd_oneframe+z
    return zadd_oneframe

def split_list(alist, wanted_parts=1):
    length = len(alist)                                                                    
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]  

def even_or_odd(num):
    if (num % 2) == 0:
        return True
    else:
        return False 

def gauss_exact(pos_traj, xpos_zregion, ypos_zregion, xedges, yedges, amp=1,a=125,b=125,c=125,res_factor=0.1,width=0.1,npad=1,
                substrate=False,water=False,iceface=False,plot2d=False,plot2d_where=False,plot3d=False,plot_conv=False,
                split_trajectory=False,traj_splits='even_int', give_grid=False, provided_gridx=None, provided_gridy=None):
    """Places gaussians at exact positions of atoms on a grid made by make_grid."""
    if give_grid == False:
        gridx, gridy, X, Y, res_width = make_grid(xedges,yedges,res_factor,width,npad)
    elif give_grid == True:
        gridx = provided_gridx
        gridy = provided_gridy
        res_width = np.diff(gridx[:,0])[0]

    xyi = np.vstack([gridx.ravel(), gridy.ravel()]) #this is grid passed to gauss2d function
    ###1st if is to make gauss surface for substrate###
    if substrate == True and water == False:
        conv_factor = 1 # by definition
        zadd = [] #initialise list of guassians plotted                                     
        for i in np.arange(len(xpos_zregion)):                                              
            if len(xpos_zregion) != len(ypos_zregion):
                raise Exception("***ERROR***There are not equal number for x and y points. This does not make sense as xpos_zregion and ypos_zregion should be (x0,x1,..),(y0,y1,..) for atom positions ((x0,y0),(x1,y1)..)")
            else:                                                                           
                z = gauss2d(xyi,amp,xpos_zregion[i],ypos_zregion[i],a,b,c,dx=res_width)     
                z.shape = gridx.shape                                                       
                zadd.append(z)                                                              
        zpred = sum(zadd)                                                                   
        ##checking number of molecules represented by gaussian surface                      
        m_natom = len(xpos_zregion) #only 1 frame so not proper mean                        
        zcopy = copy.deepcopy(zpred) #make copy so dont change shape of zpred               
        s1,s2 = zcopy.shape                                                                 
        zcopy.shape = s1*s2 #reshape into 1d array to integrate along                       
        int_natom=np.trapz(zcopy,dx=res_width)                                              
        print("Number of substrate molecules in frame (and region) calculated directly:", m_natom, "\nNumber of molecules from integral of gaussian surface:", int_natom)
        if round(m_natom) != round(int_natom):
            raise Exception("********ERROR********Integral of gaussian surface does not correctly reproduce number of atoms on average per frame. Must be an error")
    ###2nd if is to make gauss surface for water###                                         
                                                                                            
    if water == True and substrate == False:                                                
        print("Placing gaussians for water trajectory.. \nReading", len(xpos_zregion), "frames")
        if split_trajectory==True:                                                          
            print("Splitting trajectory into", traj_splits, "parts. To fix memory leak for large systems\n")
            start_time=time.time()                                                          
            input_list=np.array_split(np.arange(len(xpos_zregion)),traj_splits) #split inputs
            zpred=[] #so if statement definitley is hit                                     
            zpred_list=[]                                                                   
            for in_index in np.arange(len(input_list)):                                     
                if even_or_odd(traj_splits)==False:                                         
                    raise Exception("*********BREAK ERROR********trajectory must be split by even number so can combine properly to calculate convergence factor. Pass even integer.")
                elif even_or_odd(traj_splits)==True:                                        
                    inputs=input_list[in_index]                                             
                    print("Reading frames", inputs[0], "to", inputs[-1])
                    #zpred_longlist is one grid for every frame                             
                    #zpred_list here is now a storage for a whole split (remove RAM overhead)
                    #..as only stores traj_splits long list. Instead of frames long list of grids
                    zpred_longlist = Parallel(n_jobs=-1,backend='threading')(delayed(gauss_frame_RAM)(i,xyi,amp,a,b,c,res_width,gridx, xpos_zregion, ypos_zregion) for i in inputs)
                    if len(zpred)==0:                                                       
                        zpred=sum(zpred_longlist)                                           
                        zpred_list.append(zpred)                                            
                    else:                                                                   
                        zpred=zpred+sum(zpred_longlist)                                     
                        zpred_list.append(sum(zpred_longlist))                              
            zpred=np.array(zpred)/len(xpos_zregion) #normalise by number of frames          
            print("\nAfter splitting trajectory, program took", time.time()-start_time, "to place gaussians\n")
                                                                                            
        if split_trajectory==False:                                                         
            print("Running parrallel over full trajectory. Note should split if large area system due to very slow memory leak of storing zpred for each frame.")
            start_time=time.time()                                                          
            inputs = np.arange(len(xpos_zregion)) #frames                                   
            zpred_list = Parallel(n_jobs=-1,backend='threading')(delayed(gauss_frame_RAM)(i,xyi,amp,a,b,c,res_width,gridx, xpos_zregion, ypos_zregion) for i in inputs)
            zpred = sum(zpred_list)                                                         
            zpred = zpred/len(xpos_zregion) #normalise by number of frames                  
            print("Program took", time.time()-start_time, "to place gaussians")
        ##checking number of molecules represented by gaussian surface directly
        natom=[]                                                                            
        for i,val in enumerate(xpos_zregion):                                               
            natom.append(len(val))                                                          
        m_natom=np.mean(natom)                                                              
        #integrating surface                                                                
        zcopy = copy.deepcopy(zpred) #make copy so dont change shape of zpred               
        s1,s2 = zcopy.shape                                                                 
        zcopy.shape = s1*s2 #reshape into 1d array to integrate along                       
        int_natom=np.trapz(zcopy,dx=res_width) #only provide y values, not x, so important dx is     
        #..correctly assigned to be the actual gap between molecules                        
        print("Mean number of water molecules per frame calculated directly:", m_natom, "\nNumber of molecules from integral of gaussian surface:", int_natom)
        if round(m_natom) != round(int_natom):
            raise Exception("********ERROR********Integral of gaussian surface does not correctly reproduce number of atoms on average per frame. Must be an error")
        ##convergence test                                                                  
        x1=split_list(zpred_list,wanted_parts=2)[0] #sum(x1) is one grid                    
        x2=split_list(zpred_list,wanted_parts=2)[1]                                         
        conv_factor = np.sum(abs((sum(x1) - sum(x2))))/np.sum(x1) #MUST be absolute sum     
        #..of the differences you take. Otherwise -ve and +ve cancel out                    
        print("convergence factor for water gaussian surface is:", conv_factor)
        if plot_conv == True:                                                               
            ##N.B. plotted split histograms on same                                         
            #plt.clf()                                                                      
            plt.rc('font', size= 15)                                                        
            fig = plt.figure()                                                              
            ax = fig.add_subplot(131,adjustable='box-forced')                               
            ax.set(adjustable='box-forced',aspect='equal')                                  
            ax.set_title("1st half data")                                                   
            vmin = np.amin(sum(x1))                                                         
            vmax = np.amax(sum(x1))                                                         
            p1 = ax.scatter(gridx,gridy,c=sum(x1), marker='x',vmin=vmin, vmax=vmax)         
            colorbar(p1)                                                                    
            ax = fig.add_subplot(132,adjustable='box-forced')                               
            ax.set(adjustable='box-forced', aspect='equal')                                 
            ax.set_title("2nd half data")                                                   
            p2 = ax.scatter(gridx,gridy,c=sum(x2), marker='x', vmin=vmin, vmax=vmax)        
            colorbar(p2)                                                                    
            ax = fig.add_subplot(133,adjustable='box-forced')                               
            ax.set(adjustable='box-forced', aspect='equal')                                 
            ax.set_title('1st minus 2nd\n Convergence factor = {}'.format(conv_factor))
            p3 = ax.scatter(gridx,gridy,c=(sum(x2)-sum(x1)), marker='x', vmin=vmin, vmax=vmax)
            colorbar(p3)                                                                    
            plt.show()                                                                      
    if water == False and substrate == False:                                               
        print("*****ERROR***** - Passed incorrect arguements to gauss_exact. Must tell it to work for either water or substrate - told it to work for neither")
    if plot2d == True:                                                                      
        ##plotting gaussgrid                                                                
        plt.clf()                                                                           
        ##uncomment to see where put gaussians each time - lot of points for water!         
        if plot2d_where == True:                                                            
            if water == True and substrate == False:                                        
                xscatter, yscatter = OLD_zregionextract_xypos_water(z_list,waterpos_traj)   
                plt.scatter(xscatter,yscatter, marker='D', s=100, facecolors='none', edgecolors='b')
            if substrate == True and water == False:                                        
                plt.scatter(xpos_zregion,ypos_zregion, marker='D', s=100, facecolors='none', edgecolors='b')
        plt.scatter(gridx,gridy, marker='x',c=zpred,zorder=0)                               
        plt.colorbar()                                                                      
        plt.show()                                                                          
    if plot3d == True:                                                                      
        plt.clf()                                                                           
        s1,s2 = zpred.shape                                                                 
        zpred.shape = s1*s2                                                                 
        fig = plt.figure()                                                                  
        ax = fig.add_subplot(111, projection='3d')                                          
        p = ax.scatter(gridx,gridy,zpred,depthshade=True, marker='x',c=zpred)               
        fig.colorbar(p)                                                                     
        plt.show()                                                                          
                                                                                            
    if iceface == True:                                                                     
        return zpred, gridx, gridy                                                          
    else:                                                                                   
        return zpred, gridx, gridy, conv_factor                                             

def my_roll(grid,n_up,n_side,up=False,side=False):                                       
    """function to do np.roll properly to the grid.                      
    np.roll by itself wraps slightly off"""                              
    ptest=[]                                                             
    if up == True and side == False:                                     
        for i,val in enumerate(grid):                                    
            ptest.append(np.roll(val,n_up))                        
        ptest=np.array(ptest)                                      
    if up == False and side == True:                                                      
        ptest=np.roll(grid,n_side,axis=0)                                       
    if up == True and side == True:                                     
        for i,val in enumerate(grid):                              
            ptest.append(np.roll(val,n_up))                                   
        ptest=np.array(ptest)                                                   
    ptest=np.roll(ptest,n_side,axis=0)                                          
    return ptest                                                                

def integrate_gauss(zpred,res_width):
    zcopy = copy.deepcopy(zpred) #make copy so dont change shape of zpred               
    s1,s2 = zcopy.shape
    zcopy.shape = s1*s2 #reshape into 1d array to integrate along                       
    int_natom=np.trapz(zcopy,dx=res_width)
    return int_natom

def plot_gauss(zpred,gridx,gridy,plot2d=False,plot3d=False):
    if plot2d == True:
        plt.clf()
        plt.scatter(gridx,gridy, marker='x',c=zpred,zorder=0)
        plt.colorbar()
        plt.show()
    if plot3d == True:
        plt.clf()
        s1,s2 = zpred.shape
        zpred.shape = s1*s2
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(gridx,gridy,zpred,depthshade=True, marker='x',c=zpred)
        fig.colorbar(p)
        plt.show()

def split_grid_squares(im,N):
    """split grid into TWO N parts
    Returns a LIST of arrays that
    are square slices of the image"""
    grid_split=[]
    if even_or_odd(N) == False or N<4:
        raise Exception("\n\n***ERROR***Must split the grid into even number of tiles and must be at least 4 \n\n")
    #split veritcally:
    n = N/2
    for i in np.arange(n): #loop over each veritcal slice
        v_slice=np.array_split(im,n,axis=0)[i]
        #slice the vertical slice horiztontally
        square_slice = np.array_split(v_slice,n,axis=1)
        #print "\nHERE\n", square_slice, len(square_slice)
        ##slices can be different sizes - so need to make array of different sized arrays
        for it0 in np.arange(len(square_slice)):
            tmp=np.array(square_slice[it0])
            grid_split.append(tmp)
            tmp=[]
    return grid_split

def cutempty_x(zpred_foranx,tolerance):
    nonzero_foranx=[]                                    
    for i,val in enumerate(zpred_foranx):                                                  
        if val > tolerance:
            nonzero_foranx.append(val)                                                     
    return nonzero_foranx                                                                  
                                                                                           
                                                                   
def cont_1st_last_indices(array):           
    """Function that takes first continuous region of array and takes last continuous region of array. To be used to cut empty grid space from zpred,gridx,gridy """
    redun_1st=[]
    array_1st=[]                                                          
    for i in np.arange(len(array)):
        if i +1 < len(array):                                             
            if array[i] == array[i+1] -1:                                 
                redun_1st.append(i)
                array_1st.append(array[i])                                                 
            else:
                redun_1st.append(i) #gets end of continuous
                array_1st.append(array[i])                                                 
                break                                                                      
    #go from last index to start to find last redundant area               
    redun_last=[]
    array_last=[]        
    for i in np.absolute(np.arange(-len(array)+1,1)):
        if i +1 < len(array):
            if array[i] == array[i+1] -1:                                     
                redun_last.append(i+1)
                array_last.append(array[i+1])                                              
            else:
                redun_last.append(i+1) #gets end of continuous                
                array_last.append(array[i+1])                                              
                break                                                                      
    #array_last reverse order        
    return array_1st, array_last[::-1] 

def trim_grid(zpred_norot,zrot,zcut,plot=False):                                           
    """Fucntion that trims the grids - useful for non orthomrhombic simulation cells - where when you rotate (since i dont reshape the grid) has large areas of empty space over which the ice cut must run - greatly reduces cost as at least N^2 dependency on grid dimensions
    Inputs:                                                                                
    zpred_norot - original gaussian surface from system                                    
    zrot - rotated gaussian surface                                                        
    zcut - ice face being placed                                                           
    """                                                                                    
                                                                                           
    redundant_x=[]                                                                         
    for i in np.arange(zpred_norot.shape[0]):
        if len(cutempty_x(zrot[i],tolerance=10**-3)) == 0:                                 
            redundant_x.append(i)                                                          
                                                                                           
    cut_1st, cut_last = cont_1st_last_indices(redundant_x)                                 
    #print "cutting specifics", len(cut_1st), len(cut_last)                                
    ##Checking the start values of cut_1st and cut_last - must be from start and end of x axis 
    if len(cut_1st) != 0: #needed to avoid errors for no trim results                        
        if cut_1st[0] != 0:        
            print("Not trimming from start as redundant section doesnt start from index 0, starts from", cut_1st[0],
                  "which would cut non redundant start section")
            cut_1st=[] #then set cut_1st to empty so second if loops can deal with it        
    if len(cut_last) != 0:        
        if cut_last[-1] != (zpred_norot.shape[0] -1):
            print("Not trimming from end as redundant section doesnt start from last index", (zpred_norot.shape[0]-1),
                  "starts from", cut_last[-1], "which would cut non redundant end section")
            cut_last=[]

    #Second if statements     
    if len(cut_1st) != 0 and len(cut_last) != 0: #cut at start and end                     
        gridx_trim=gridx[cut_1st[-1]:cut_last[0]]                                          
        gridy_trim=gridy[cut_1st[-1]:cut_last[0]]                                         
        zrot_trim=zrot[cut_1st[-1]:cut_last[0]]                                            
    if len(cut_1st) != 0 and len(cut_last) == 0: #only cut at start                        
        gridx_trim=gridx[cut_1st[-1]:-1]                                                  
        gridy_trim=gridy[cut_1st[-1]:-1]                                                   
        zrot_trim=zrot[cut_1st[-1]:-1]                                                     
    if len(cut_1st) == 0 and len(cut_last) != 0: #only cut at end                          
        gridx_trim=gridx[0:cut_last[0]]                                          
        gridy_trim=gridy[0:cut_last[0]]                                          
        zrot_trim=zrot[0:cut_last[0]]                                            
    if len(cut_1st) == 0 and len(cut_last) == 0: #no cutting                               
        gridx_trim=copy.deepcopy(gridx)                                                    
        gridy_trim=copy.deepcopy(gridy)                                                    
        zrot_trim=copy.deepcopy(zrot)                                                      
                                                                                           
    if plot==True:                                                                         
        plt.scatter(gridx_trim,gridy_trim, marker='x',c=zrot_trim,zorder=0)                
        plt.colorbar()                                                                     
        plt.show()                                                                         
                                                                                           
    ##now calculate new roll list for the trimmed grid##                                   
    #..calculating positions available to put ice face                                     
    #ice face must fit-i.e. can't place part of it off of the grid for the system (gridx)  
    s1,s2 = zrot_trim.shape                                                                
    t1,t2 = zcut.shape                                                                     
    ex_max=s1-t1                                                                           
    ey_max=s2-t2                                                                           
    roll_list=list(itertools.product(np.arange(ex_max),np.arange(ey_max)))                 
                                                                                           
    return gridx_trim, gridy_trim, zrot_trim, roll_list

def zpred_squarecut(zpred,zcut,limit=450):
    """########ZPRED SQUARE CUT#######
    ###cutting zpred to be square with largest possible density enclosed###
    ###AND to have max size of 4.5 nm thus increasing speed of calculations
    for large systems###
    Limit: 100 = 1 nm
    Inputs: zpred - the gaussian surface to make square cut from
    zcut - the ice face to make new roll_list from"""
    ##square dimensions:
    if np.min(zpred.shape) < limit:
        print("Not square limited")
        #then smallest side of zpred rectangle is less than 4.5 so have square dim based off of that 
        square_dim=np.min(zpred.shape) #equal smallest side of zpred rectangle                       
        #extract which dimension was the shortes                                                     
        x_or_y=np.argmin(zpred.shape) #0 = x, 1 = y                                                  
        limited=False
    else: #set square size myself                                             
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
            #print "x dimension was shortest of zpred rectangle - hence taking as maximum of square cut dimension. Move the suqare along y direction to find max density location."
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
    else:
        raise Exception("***ERROR***x_or_y not determined")

    ##now move the square along and take the maximum density position
    density_list=[]
    for i,val in enumerate(roll_list):
        edge_coordinate = (val[0],val[1])
        slicer = tuple(slice(edge, edge+i) for edge, i in zip(edge_coordinate, (square_dim,square_dim)))
        density_list.append(np.sum(zpred[slicer]))
    #find max position                                                                                  
    max_position=np.argmax(density_list)
    density_list=[] #remove the unecessary data list (for RAM) 
    #make the actual max placement tuple                            
    if x_or_y == 0:
        if limited == False:
            max_tuple=(0,max_position)
        if limited == True:
            max_tuple=(int(ex_max/2),max_position)
    elif x_or_y == 1:
        if limited == False:
            max_tuple=(max_position,0)
        if limited == True:
            max_tuple=(max_position,int(ey_max/2))
    else:
        raise Exception("***ERROR***x_or_y not determined")
    #make the slice for the max placement         
    slicer = tuple(slice(edge, edge+i) for edge, i in zip(max_tuple, (square_dim,square_dim)))

    ##now calculate new roll list for the trimmed grid##                                      
    #..calculating positions available to put ice face                                        
    #ice face must fit-i.e. can't place part of it off of the grid for the system (gridx)     
    s1,s2 = zpred[slicer].shape
    t1,t2 = zcut.shape
    ex_max=s1-t1
    ey_max=s2-t2
    roll_list=list(itertools.product(np.arange(ex_max),np.arange(ey_max)))

    ###Plotting the new square cut zpred - use to check##                                      
    #fig = plt.figure(figsize=(20,10))                                                             
    #ax = fig.add_subplot(121,adjustable='box-forced')                                             
    #ax.set(adjustable='box-forced',aspect='equal')                                            
    #ax.set_title("Gaussian surface")                                                          
    #p1 = ax.scatter(gridx,gridy,c=zpred, marker='x')                                          
    #colorbar(p1)                                                                              
    #ax = fig.add_subplot(122,adjustable='box-forced')                                         
    #ax.set(adjustable='box-forced',aspect='equal')                                        
    #ax.set_title("Gaussian surface square cut")                                           
    #p1 = ax.scatter(gridx[slicer],gridy[slicer],c=zpred[slicer], marker='x')              
    #colorbar(p1)                                                                         
    #plt.show()  

    return gridx[slicer], gridy[slicer], zpred[slicer], roll_list

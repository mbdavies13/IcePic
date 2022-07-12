"""
Script used to make xy images of water contact layers.
Reads in trajectory and places gaussians at atoms positions.
Contact layer is averaged over frames.

Appliying to full database

Default is not to assume a structure of the database.
But if have same structure as used in the paper (see the Apollo repository)
then much fewer args are needed for the inputs and outputs -- this is done below.
- See make_xy_water() in general for more info
"""
import numpy as np
from general.make_xy import make_xy_water

# Decide how to split up how many times to split up the trajectory
# Reduces RAM overhead -- list storage of frames used to enable testing convergence
TRAJ_SPLITS = 2 # must be an even number

# PATH = path to structure directory
# - overidden if provide explicit args for e.g. topoFILE, trjFILE (see make_xy_water())
# note: trajectories not in Apollo rep due to memory requirement
# .. but lammps input files to recreate are present
PATH = '../../structures' # PATH TO TRAJECTORIES

#######################################
# read in system names and indexes [[1, n1], [2,n2], ..., [N, nN]]
sys_list = np.genfromtxt('../NameList.txt', dtype='str')

# loop through systems and make imags
for sys_i, sysname in sys_list[:1]:
    print('\n\n####### {0} {1}\n'.format(sys_i, sysname))
    make_xy_water(sysname, TRAJ_SPLITS,
                  path=PATH,
                  show_plots=True,
                  just_pz=True # use if just want to see pz
                  )

"""
Script used to make xy images of water contact layers.

Simple example for the struct and trajectory in LJ_fcc111_a1e1/

"""
import numpy as np
from general.make_xy import make_xy_water

# Decide how to split up how many times to split up the trajectory
# Reduces RAM overhead -- list storage of frames used to enable testing convergence
TRAJ_SPLITS = 2 # must be an even number


# PATH = path to structure directory
PATH = '../'
sysname = 'LJ_fcc111_a1e1'

make_xy_water(sysname, TRAJ_SPLITS,
              trjFILE=PATH+sysname+'/lammps/toy_traj.xtc',
              path=PATH,
              show_plots=True,
              just_pz=False # use if just want to see pz
              )

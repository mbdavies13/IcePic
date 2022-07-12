"""
Script used to make contact layer images of KAO plus TIP4P system
See Apollo repository to get extract files and run trajectory if wish to recreate
"""
from general.make_xy import make_xy_water

### Variables need too pass - enter values required
# TRAJ_SPLITS =
# path =
# topoFILE =
# trjFILE =
# zpredFILE =
# sysname =

#########
# # make xy water image - of I exposed surface
make_xy_water(sysname=sysname,
              TRAJ_SPLITS=TRAJ_SPLITS,
              atom_selection='name OW',
              path=path,
              topoFILE=topoFILE,
              trjFILE=trjFILE,
              boxarea=3225.73,
              show_plots=True,
              # name of file when save
              zpredFILE=zpredFILE
              )
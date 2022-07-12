"""
Script used to make contact layer images of AgINaCl plus TIP4P system
See Apollo repository to get extract files and run trajectory if wish to recreate

Note: since two interface system must use z_limit and filp args in make_xy_water()
"""
from general.make_xy import make_xy_water

### Variables need too pass - enter values required
# TRAJ_SPLITS =
# path =
# sysname =
# topoFILE =
# trjFILE =
# zpredFILE =
# sysname =


#########
# make xy water image - of I exposed surface
make_xy_water(sysname, TRAJ_SPLITS,
              atom_selection='name 1 or name 4 or name 5',
              path=path,
              topoFILE=topoFILE,
              trjFILE=trjFILE,
              show_plots=True,
              z_limit=6.0, # needed so doesnt include top interface in pz
              # name of file when save
              zpredFILE=zpredFILE
              )

## make xy water image - of Ag exposed surface
# This is top interface in z
# Code takes interfaces assuming substrate on bottom
# Thus, use flip=True to flip system so Ag surface is at bottom
make_xy_water(sysname, TRAJ_SPLITS,
              atom_selection='name 1 or name 4 or name 5',
              path=path,
              topoFILE=topoFILE,
              trjFILE=trjFILE,
              show_plots=True,
              z_limit=-6.0, # needed so doesnt include top interface in pz
              flip=True,
              zpredFILE=zpredFILE
)
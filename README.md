# IcePic
![Alt text](./IcePic_logo.jpg?raw=true "Title")

Scripts for creating, measuring and interpreting the IcePic model for prediction of nucleation temperatures.

This repository contains the code used to support the findings of the paper:

"Accurate prediction of ice nucleation from room temperature water" Michael Benedict Davies, Martin Fitzner, Angelos Michaelides.

The simulation inputs, data and code are also available at: https://doi.org/10.17863/CAM.81078

## License
The content of this repository is licensed under the CC-BY-NC-SA-4.0 license. See the file `LICENSE` for details.

## Contents:
* `general`: module containing utility functions of general use.
* `models`: module containing functions to build convolutional neural networks and dummy models
* `do_xygauss`: create images of water contact layers from simulation trajectories
  * `LJ_fcc111_a1e1`: inputs of an example structure upon which the creation of a water contact layer is demonstrated in `do_xygauss`. Note, this structure is included just for demonstration purposes and will not create a converged water contact layer. See the paper as well as https://doi.org/10.17863/CAM.81078 for full simulation details and inputs.
* `do_regression`: train convolutional neural networks that make up IcePic.
* `do_ensemble`: measure the performance of the ensembles of neural networks that make up IcePic.
* `do_dummy`: measure the performance of dummy models.
* `do_reverse_interp`: apply reverse interpretation methods to IcePic.

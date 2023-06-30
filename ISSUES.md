## Open Issues
* Change references from link format to ACA format
* Fix parameters not being loaded into launch files & need to add arguments that can be loaded from cmdline to node
* Fix the MPC to work with the cartpole simulation (Set up threaded controller based of timer for mpc)

## Next steps
* Another demonstration
* Koopman?

## Future features
* Custom objective functions and directional derivative functions
* Setup codespaces for online testing 

## WORKING ISSUES
* Change the controllers (ilQR / Ergodic) to use a base class - several functions are shared
* NEED TO MOVE CARTPOLE.hpp && Test files

## Test Issues
* Move the fourier basis and measurement initialization into the controller - easier interface
* Move files related to cartpole and running cartpole controller to the cartpole package

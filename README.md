# RayleighMC

A toy MC program for Rayleigh scattering angular / polarisation simulation.

## Implementations:
- Rayleigh_class : a class describing Rayleigh scattering for single photon. 
                   Polarisation and momentum of the incident photon need to be specified. 
                   
                   1. If no scattering direction is specified, sampling the scatterred photon momentum and polarisation;
                   
                   2. If scattered light directionn is fixed, calculating the polarisation direction.

- detector: describing the physical properties of an ideal detector. The most important one is the detection polarisation direction.

## Tools:
- vector_method.py : return perpendicular vector
- calculateRotationMatrix.py : some matrix calculation

## Usage:
- example1.py : for isotropic molecules. Detect scatterred light at certain direction with polariser.
- example2.py : for anisotropic molecules. Detect scatterred light at certain direction with polariser. Validate Hh angular distribution.
- example3.py : at Phi=270 deg, check if the rho_v == Hv/Vv for different input rhov.
- example4.py : plan to compare MC sim and measurement data.


## Potential Problems:
1. Normalization of polarisability tensor

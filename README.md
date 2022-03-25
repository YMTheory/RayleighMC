# RayleighMC

\textbf{Author: Miao Yu -> miaoyu@whu.edu.cn}

MC program for angular / polarisation simulation of Rayleigh scattered photons of anisotropic molecules.

Python scripts have been implemented and validated for toy study and algorithm development.

A simple Geant4 package has been implemented in dir "./Ray".

## Implementations:
-PYTHON:
  - Rayleigh_class : a class describing Rayleigh scattering for single photon. 
                     
    - Polarisation and momentum of the incident photon need to be specified. 
                     
    - Depolarization ratio is required as a material property for calculation.
                   
    - Functions: 

      1) calculatePol / calculatePol_modified: given certain scattering direction before, calculate polarization of scattered photons.
      
      2) sampleMomPol / sampleMomPol_modified: sample momentum and polarization of scattered photons.

  - detector: describing the physical properties of an ideal detector. The most important one is the detection polarisation direction by using "set_detPol()".


- Geant4:
  Geant4 codes are implemented in dir "Ray/" where a Geant4.10.06 version software has been used.
  
  - geometry: a simple sample was placed at center with customed Rayleigh scattering length (RAYLEIGH) and penpendicularly polarized depolarization ratio (RHOV); A 4Pi sensitive detector surrounds the sample outside to detect photons and give the angular/polarization information.
  - Optical process: a customed RayleighScattering class has been implemented to sample scattered photons for anisotropic liquids. (MatrixCalc class has been added for matrix and vector calcualtion in new class);
  - analyser: save root files of simulation fo analysis.


## Tools for matrix calculation:
- vector_method.py : return perpendicular vector
- calculateRotationMatrix.py : some matrix calculation

## Usage:
- example1.py : for isotropic molecules. Detect scatterred light at certain direction with polariser.
- example2.py : for anisotropic molecules. Detect scatterred light at certain direction with polariser. Validate Hh angular distribution.
- example3.py : at Phi=270 deg, check if the rho_v == Hv/Vv for different input rhov.
- example4.py : compare MC sim and measurement data.
- example5.py : sample scattered photons of anisotropic liquids.

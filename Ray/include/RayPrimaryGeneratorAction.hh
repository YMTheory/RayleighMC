//
// $Id: RayPrimaryGeneratorAction.hh 90623 2015-06-05 09:24:30Z gcosmo $
//
/// \file RayPrimaryGeneratorAction.hh
/// \brief Definition of the RayPrimaryGeneratorAction class

#ifndef RayPrimaryGeneratorAction_h
#define RayPrimaryGeneratorAction_h 1

#include "RayAnalysisManager.hh"

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;

/// The primary generator action class with particle gun.
///
/// The default kinematic is a 6 MeV gamma, randomly distribued 
/// in front of the phantom across 80% of the (X,Y) phantom size.

class RayPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    RayPrimaryGeneratorAction();    
    virtual ~RayPrimaryGeneratorAction();

    // method from the base class
    virtual void GeneratePrimaries(G4Event*);         
  
    // method to access particle gun
    const G4ParticleGun* GetParticleGun() const { return fParticleGun; }
  
  private:
    G4ParticleGun*  fParticleGun; // pointer a to G4 gun class
    G4int NumberOfParticlesToBeGenerated;

  private:
    RayAnalysisManager* analysis;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

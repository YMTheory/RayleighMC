// $Id: RayPrimaryGeneratorAction.cc 94307 2015-11-11 13:42:46Z gcosmo $
//
/// \file RayPrimaryGeneratorAction.cc
/// \brief Implementation of the RayPrimaryGeneratorAction class

#include "RayPrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <cmath>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayPrimaryGeneratorAction::RayPrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction()
{
    fParticleGun      = new G4ParticleGun();

    analysis          = RayAnalysisManager::getInstance();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayPrimaryGeneratorAction::~RayPrimaryGeneratorAction()
{
  delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RayPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    

    
    // Generate more than one particle each time
    G4int NumberOfParticlesToBeGenerated = 1;
    fParticleGun = new G4ParticleGun(NumberOfParticlesToBeGenerated);

    // particle definition
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4String particleName;
    //G4ParticleDefinition* particle = particleTable->FindParticle(particleName="e-");
    G4ParticleDefinition* particle = particleTable->FindParticle(particleName="opticalphoton");


    // set particle type
    fParticleGun->SetParticleDefinition(particle);

    // set optical photon energy/wavelength
    fParticleGun->SetParticleEnergy(2*eV);

    // set momentum direction
    G4double mom_x, mom_y, mom_z ;
    mom_x = 0;
    mom_y = 0;
    mom_z = 2*eV;
    fParticleGun->SetParticleMomentumDirection( G4ThreeVector(mom_x, mom_y, mom_z));

    G4double pol_x, pol_y, pol_z;
    pol_x = 1;
    pol_y = 0;
    pol_z = 0;
    fParticleGun->SetParticlePolarization( G4ThreeVector(pol_x, pol_y, pol_z) );
    analysis->analyseInitPolX(pol_x);
    analysis->analyseInitPolY(pol_y);
    analysis->analyseInitPolZ(pol_z);

    fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., 0.));

    fParticleGun->GeneratePrimaryVertex( anEvent );
     

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


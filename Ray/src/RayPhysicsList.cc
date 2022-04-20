/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Fri Sep 25 13:36:45 2020
 @File Name: RayPhysicsList.cc
 ************************************************************************/

#include "G4ProcessManager.hh"
#include "RayPhysicsList.hh"
#include "RayleighScattering.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4HadronPhysicsQGSP_BERT_HP.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4IonPhysicsPHP.hh"

#include "G4OpticalPhysics.hh"
#include "G4SystemOfUnits.hh"

// particles
#include "G4LeptonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

#include "G4OpticalPhoton.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"


RayPhysicsList::RayPhysicsList() : G4VModularPhysicsList()
{
    defaultCutValue = 1.0*mm;

    RegisterPhysics(new G4EmLivermorePhysics());

}

RayPhysicsList::~RayPhysicsList() {
}

void RayPhysicsList::SetCuts() {
    SetCutsWithDefault();
}


void RayPhysicsList::ConstructParticle()
{

    G4OpticalPhoton::OpticalPhotonDefinition();

    G4Gamma::GammaDefinition();
    G4Electron::ElectronDefinition();
    G4Positron::PositronDefinition();

}

void RayPhysicsList::ConstructProcess()
{
    AddTransportation();
    ConstructOpticalProcess();
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"

void RayPhysicsList::ConstructOpticalProcess()
{
    G4OpAbsorption* theAbsProcess         = new G4OpAbsorption();
    RayleighScattering* theRayProcess     = new RayleighScattering();
    //G4OpRayleigh* theRayProcess           = new G4OpRayleigh();
    G4OpBoundaryProcess* theBdProcess     = new G4OpBoundaryProcess();
    theBdProcess->SetInvokeSD(false);
    auto particleIterator = GetParticleIterator();
    particleIterator->reset();
    while( (*particleIterator)() ){

        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        if (theAbsProcess->IsApplicable(*particle)) {
            pmanager -> AddDiscreteProcess(theAbsProcess);
            G4cout << " ===> Registered Absorption Process " << G4endl;
        }

        if(theRayProcess->IsApplicable(*particle)) {
            pmanager -> AddDiscreteProcess(theRayProcess);
            G4cout << " ===> Registered Rayleigh Scatterinng Process " << G4endl;
        }

        if(theBdProcess->IsApplicable(*particle)) {
            pmanager -> AddDiscreteProcess(theBdProcess);
            G4cout << " ===> Registered Boundary Process " << G4endl;
        }
    }
}

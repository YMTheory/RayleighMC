#include "RaySteppingAction.hh"
#include "RayEventAction.hh"

#include "G4Track.hh"
#include "G4ThreeVector.hh"
#include "G4StepPoint.hh"
#include "G4Step.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

RaySteppingAction::RaySteppingAction(RayEventAction* event)
: G4UserSteppingAction()
{;}

RaySteppingAction::~RaySteppingAction()
{;}

void RaySteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4StepPoint* postStep = step->GetPostStepPoint();
    G4cout << track->GetTrackID() << " " << track->GetParentID() << " "  
           << track->GetKineticEnergy()*MeV << " " 
           << track->GetParticleDefinition()->GetParticleName() << " "  
        //<< G4endl;
           << postStep->GetProcessDefinedStep()->GetProcessName() << G4endl;
    

}

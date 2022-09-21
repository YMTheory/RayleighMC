#include "RaySteppingAction.hh"
#include "RayEventAction.hh"

#include "G4Track.hh"
#include "G4ThreeVector.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh"

RaySteppingAction::RaySteppingAction(RayEventAction* event)
: G4UserSteppingAction()
{;}

RaySteppingAction::~RaySteppingAction()
{;}

void RaySteppingAction::UserSteppingAction(const G4Step* step)
{
    if(0) {
        G4cout << step->GetTrack()->GetTrackID() << step->GetPreStepPoint()->GetPosition() << " "<< step->GetPreStepPoint()->GetPhysicalVolume()->GetName() << " " 
        << step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << " "
        << step->GetPostStepPoint()->GetPosition() << " "
        << step->GetPostStepPoint()->GetStepStatus() << " "
        << G4endl;

    }
}

#include "RaySteppingAction.hh"
#include "RayEventAction.hh"

#include "G4Track.hh"
#include "G4ThreeVector.hh"

RaySteppingAction::RaySteppingAction(RayEventAction* event)
: G4UserSteppingAction()
{;}

RaySteppingAction::~RaySteppingAction()
{;}

void RaySteppingAction::UserSteppingAction(const G4Step* step)
{
    //G4Track* track = step->GetTrack();
    //G4ThreeVector prePos  = step->GetPreStepPoint()->GetPosition();
    //G4ThreeVector postPos = step->GetPostStepPoint()->GetPosition();
    //G4cout << track->GetTrackID() << " " << prePos.x() << " " << prePos.y() << " " << prePos.z() << " ---> " 
    //       << postPos.x() << " " << postPos.y() << " " << postPos.z() << " " << G4endl;

}

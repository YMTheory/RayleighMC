
#include "RayActionInitialization.hh"
#include "RayPrimaryGeneratorAction.hh"
#include "RayRunAction.hh"
#include "RayEventAction.hh"
#include "RayTrackingAction.hh"
#include "RaySteppingAction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayActionInitialization::RayActionInitialization()
 : G4VUserActionInitialization()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayActionInitialization::~RayActionInitialization()
{}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RayActionInitialization::Build() const
{
  SetUserAction(new RayPrimaryGeneratorAction);

  RayRunAction* runAction = new RayRunAction();
  SetUserAction(runAction);

  RayEventAction* eventAction = new RayEventAction();
  SetUserAction(eventAction);

  SetUserAction(new RayTrackingAction);
  SetUserAction(new RaySteppingAction(eventAction));
  
}  

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

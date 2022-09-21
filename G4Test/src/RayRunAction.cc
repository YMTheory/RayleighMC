#include "RayRunAction.hh"
#include "RayAnalysisManager.hh"

RayRunAction::RayRunAction()
: G4UserRunAction()
{;}


RayRunAction::~RayRunAction()
{;}

void RayRunAction::BeginOfRunAction(const G4Run*)
{
    G4cout << "Begin of One Run" << G4endl;

    RayAnalysisManager* analysis = RayAnalysisManager::getInstance();
    analysis->book();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RayRunAction::EndOfRunAction(const G4Run* )
{
    RayAnalysisManager* analysis = RayAnalysisManager::getInstance();
    analysis->finish();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



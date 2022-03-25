#include "RayEventAction.hh"
#include "RayAnalysisManager.hh"

#include "G4RunManager.hh"
#include "G4SDManager.hh"

RayEventAction::RayEventAction()
: G4UserEventAction(),
fPmtID(-1)
{;}

RayEventAction::~RayEventAction()
{;}

void RayEventAction::PrintEventStatistics(
                                G4int pmtTrackID) const
{
  // print event statistics
  //G4cout
  //   << "       current track ID: " 
  //   << std::setw(7) << pmtTrackID <<  "ID"
  //   << G4endl;

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


void RayEventAction::BeginOfEventAction(const G4Event* evt)
{
    G4cout << "Begin of Event " << evt->GetEventID() << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RayEventAction::EndOfEventAction(const G4Event* event)
{   
    RayAnalysisManager* analysis = RayAnalysisManager::getInstance();
    G4int evtid = event->GetEventID();
    analysis->analyseEventID( evtid );

    // Get hits collections IDs (only once)
    if(fPmtID == -1)  {
        fPmtID
          = G4SDManager::GetSDMpointer()->GetCollectionID("PmtHitsCollection");
    }

    // Get hits c/ollections
    auto pmtHC = GetHitsCollection(fPmtID, event);

    if ( pmtHC->entries()>0 ) {
        // Get hit with total values
        auto pmtHit = (*pmtHC)[pmtHC->entries()-1];

        // Print per event
        auto printModulo = G4RunManager::GetRunManager()->GetPrintProgress();
        if( ( printModulo > 1 ) && ( evtid % printModulo == 0 ) ) {
            G4cout << "---> End of event: " << evtid << G4endl;     
            PrintEventStatistics( pmtHit->GetTrackID() );
        }
        analysis->analysePhotonNumber(pmtHC->entries());
    }  else  {  // no hits in SD
        analysis->analysePhotonNumber(0);
    }

    analysis->analyseAddNtupleRow();

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayDetectorHitsCollection* 
RayEventAction::GetHitsCollection(G4int hcID,
                                  const G4Event* event) const
{
  auto hitsCollection 
    = static_cast<RayDetectorHitsCollection*>(
        event->GetHCofThisEvent()->GetHC(hcID));
  
  if ( ! hitsCollection ) {
    G4ExceptionDescription msg;
    msg << "Cannot access hitsCollection ID " << hcID; 
    G4Exception("B1EventAction::GetHitsCollection()",
      "MyCode0003", FatalException, msg);
  }         

  return hitsCollection;
}    

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
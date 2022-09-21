/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Fri Sep 25 16:13:44 2020
 @File Name: RayDetectorSD.cc
 ************************************************************************/

#include "RayDetectorSD.hh"

#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"

RayDetectorSD::RayDetectorSD( const G4String& name, 
                  const G4String& hitsCollectionName)
    : G4VSensitiveDetector(name),
    fHitsCollection(NULL)
{
    collectionName.insert(hitsCollectionName);

    analysis = RayAnalysisManager::getInstance();
}

RayDetectorSD::~RayDetectorSD()
{;}


void RayDetectorSD::Initialize(G4HCofThisEvent* hce)
{
    // Create hits collection
    fHitsCollection
        = new RayDetectorHitsCollection( SensitiveDetectorName, collectionName[0]);
    
    // Add this collection in hce
    G4int hcID 
        = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
    hce->AddHitsCollection( hcID, fHitsCollection );


    // Create hits
    //for( G4int i=0; i<fNofPmts+1; i++ )  {
    //    fHitsCollection->insert(new RayDetectorHit());
    //}
}

G4bool RayDetectorSD::ProcessHits( G4Step* aStep, G4TouchableHistory*)
{
    G4double edep = aStep->GetTotalEnergyDeposit();
    G4double stepLength = aStep->GetStepLength();
    if(edep == 0. && stepLength == 0. ) return false;

    auto touchable = (aStep->GetPreStepPoint()->GetTouchable());

    // Get pmt id
    auto pmtNumber = touchable->GetReplicaNumber(1);

    //auto hit = (*fHitsCollection)[pmtNumber];
    //if ( ! hit ) {
    //    G4ExceptionDescription msg;
    //    msg << "Cannot access hit " << pmtNumber; 
    //    G4Exception("B4cCalorimeterSD::ProcessHits()",
    //            "MyCode0004", FatalException, msg);
    //}         

    RayDetectorHit* hit = new RayDetectorHit();
    G4int trackId = aStep->GetTrack()->GetTrackID();
    hit->SetTrackID(trackId);

    fHitsCollection->insert(hit);

    G4Track* track = aStep->GetTrack();
    analysis-> analyseDetPolX(track->GetPolarization().getX());
    analysis-> analyseDetPolY(track->GetPolarization().getY());
    analysis-> analyseDetPolZ(track->GetPolarization().getZ());

    G4ThreeVector pos = aStep->GetPreStepPoint()->GetPosition();    
    G4double posx = pos.x();
    G4double posy = pos.y();
    G4double posz = pos.z();
    analysis->analyseDetPosX(posx);
    analysis->analyseDetPosY(posy);
    analysis->analyseDetPosZ(posz);


    return true;
}



void RayDetectorSD::EndOfEvent(G4HCofThisEvent*)
{
  if ( verboseLevel>1 ) { 
     G4int nofHits = fHitsCollection->entries();
     G4cout << G4endl
            << "-------->Hits Collection: in this event they are " << nofHits 
            << " hits in the tracker chambers: " << G4endl;
     for ( G4int i=0; i<nofHits; i++ ) (*fHitsCollection)[i]->Print();
  }
}

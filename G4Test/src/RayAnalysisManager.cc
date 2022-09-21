
/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Tue Sep 29 16:11:41 2020
 @File Name: RayAnalysisManager.cc
 ************************************************************************/

#include "g4root.hh"

#include "RayAnalysisManager.hh"

RayAnalysisManager* RayAnalysisManager::instance = 0;

RayAnalysisManager::RayAnalysisManager()
    : outputFileName("opt_sim")
{
    G4AnalysisManager::Instance();

}

RayAnalysisManager::~RayAnalysisManager()
{
    delete instance;
    instance = 0;

    delete G4AnalysisManager::Instance();
}

void RayAnalysisManager::book()
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    //Open an output file
    man->OpenFile(outputFileName);
    man->SetVerboseLevel(1);
    man->SetFirstHistoId(1);
    man->SetFirstNtupleId(1);

    G4cout << "Open output file: " << outputFileName << G4endl;
    man->CreateNtuple("photon", "Hits Info on SD");
    man->CreateNtupleIColumn("EventID");
    man->CreateNtupleIColumn("nPhoton");
    man->CreateNtupleIColumn("nInitPhoton");
    man->CreateNtupleDColumn("InitPosY");
    man->CreateNtupleDColumn("InitPolX");
    man->CreateNtupleDColumn("InitPolY");
    man->CreateNtupleDColumn("InitPolZ");
    man->CreateNtupleDColumn("DetPolX");
    man->CreateNtupleDColumn("DetPolY");
    man->CreateNtupleDColumn("DetPolZ");
    man->CreateNtupleDColumn("DetPosX");
    man->CreateNtupleDColumn("DetPosY");
    man->CreateNtupleDColumn("DetPosZ");
    man->FinishNtuple();
    G4cout << "Created ntuple for photon counting" << G4endl;
}

void RayAnalysisManager::finish()
{
    G4cout << "Going to save ntuples" << G4endl;
    // Save histograms
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->Write();
    man->CloseFile();
}

RayAnalysisManager* RayAnalysisManager::getInstance()
{
    if (instance==0) { instance = new RayAnalysisManager(); }
    return instance;
}


void RayAnalysisManager::SetOutputFileName(G4String newName)
{
  
  outputFileName = newName;
}

void RayAnalysisManager::analyseEventID( G4int evtid )
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn( 0, evtid );
}


void RayAnalysisManager::analysePhotonNumber(G4int number)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn( 1, number );
}

void RayAnalysisManager::analyseInitPhotonNumber(G4int number)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn( 2, number );
}

void RayAnalysisManager::analyseInitPosY(G4double posy)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 3, posy);
}

void RayAnalysisManager::analyseInitPolX(G4double polx)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 4, polx);
}

void RayAnalysisManager::analyseInitPolY(G4double poly)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 5, poly);
}

void RayAnalysisManager::analyseInitPolZ(G4double polz)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 6, polz);
}

void RayAnalysisManager::analyseDetPolX(G4double polx)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 7, polx);
}

void RayAnalysisManager::analyseDetPolY(G4double poly)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 8, poly);
}

void RayAnalysisManager::analyseDetPolZ(G4double polz)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 9, polz);
}


void RayAnalysisManager::analyseDetPosX(G4double posx)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 10, posx);
}

void RayAnalysisManager::analyseDetPosY(G4double posy)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 11, posy);
}

void RayAnalysisManager::analyseDetPosZ(G4double posz)
{
    G4AnalysisManager* man = G4AnalysisManager::Instance();
    man->FillNtupleDColumn( 12, posz);
}



void RayAnalysisManager::analyseAddNtupleRow()
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->AddNtupleRow();
}


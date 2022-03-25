/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Mon Sep 28 18:43:11 2020
 @File Name: RayTrackingAction.cc
 ************************************************************************/

#include "RayTrackingAction.hh"

#include "G4Track.hh"

RayTrackingAction::RayTrackingAction()
    : G4UserTrackingAction()
{}

RayTrackingAction::~RayTrackingAction()
{}

void RayTrackingAction::PreUserTrackingAction( const G4Track* track )
{
    //G4cout << track->GetTrackID() << " " << track->GetParentID() << G4endl;
    //if(track->GetCreatorProcess()){
    //    const G4VProcess* proc = track->GetCreatorProcess();
    //    G4cout << track->GetTrackID() << " " << track->GetParticleDefinition()->GetParticleName() 
    //       << " " << G4endl;
    //}
}

void RayTrackingAction::PostUserTrackingAction ( const G4Track* ) 
{;}

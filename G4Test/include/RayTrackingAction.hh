
/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Mon Sep 28 18:40:39 2020
 @File Name: RayTrackingAction.hh
 ************************************************************************/

#ifndef RayTrackingAction_h
#define RayTrackingAction_h 1

#include "G4UserTrackingAction.hh"
#include "globals.hh"

class RayTrackingAction : public G4UserTrackingAction {

    public:
        RayTrackingAction   ();
        ~RayTrackingAction  ();

        void PreUserTrackingAction  (const G4Track* track);
        void PostUserTrackingAction (const G4Track*);
};

#endif

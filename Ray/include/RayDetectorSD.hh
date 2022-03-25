/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Fri Sep 25 15:48:19 2020
 @File Name: B1TrackerSD.hh
 ************************************************************************/

#ifndef RayDetectorSD_h
#define RayDetectorSD_h 1

#include "G4VSensitiveDetector.hh"
#include "RayDetectorHit.hh"
#include "RayAnalysisManager.hh"

class G4Step;
class G4HCofThisEvent;

class RayDetectorSD : public G4VSensitiveDetector
{
    public:
        RayDetectorSD(const G4String& name,
                const G4String& hitsCollectionName);
        virtual ~RayDetectorSD();

        // methods from base class
        virtual void   Initialize  (G4HCofThisEvent* hitCollection);
        virtual G4bool ProcessHits (G4Step* step, G4TouchableHistory* history);
        virtual void   EndOfEvent  (G4HCofThisEvent* hitCollection);

    private:
        RayDetectorHitsCollection* fHitsCollection;
        RayAnalysisManager* analysis;
};

#endif

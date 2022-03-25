#ifndef RayEventAction_h
#define RayEventAction_h 1

#include "G4UserEventAction.hh"
#include "RayDetectorHit.hh"
#include "globals.hh"

class RayEventAction : public G4UserEventAction
{
    public:
        RayEventAction();
        virtual ~RayEventAction();

        virtual void BeginOfEventAction(const G4Event* event);
        virtual void   EndOfEventAction(const G4Event* event);

    private:
        // methods
        RayDetectorHitsCollection* GetHitsCollection(G4int hcID,
                                               const G4Event* event) const;
        void PrintEventStatistics(G4int pmtTrackID) const;

        // data members
        G4int fPmtID;

};

#endif

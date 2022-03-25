#ifndef RaySteppingAction_h
#define RaySteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class RayEventAction;

class RaySteppingAction : public G4UserSteppingAction
{
    public:
        RaySteppingAction(RayEventAction* event);
        virtual ~RaySteppingAction();


        virtual void UserSteppingAction(const G4Step*);

    private:
        RayEventAction* fEventAction;
};

#endif
#ifndef RayRunAction_h
#define RayRunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"

class G4Run;

class RayRunAction : public G4UserRunAction
{
    public:
        RayRunAction();
        virtual ~RayRunAction();

        virtual void BeginOfRunAction(const G4Run*);
        virtual void   EndOfRunAction(const G4Run*);
        

};

#endif
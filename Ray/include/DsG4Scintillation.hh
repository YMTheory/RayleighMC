#ifndef DsG4Scintillation_h
#define DsG4Scintillation_h

#include "G4VRestDiscreteProcess.hh"
#include "globals.hh"
#include "Randomize.hh"
#include "G4Poisson.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleMomentum.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4VRestDiscreteProcess.hh"
#include "G4OpticalPhoton.hh"
#include "G4DynamicParticle.hh"
#include "G4Material.hh" 
#include "G4PhysicsTable.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"

class DsG4Scintillation : public G4VRestDiscreteProcess
{

    public:
        DsG4Scintillation(const G4String& processName = "Scintillation",
                                  G4ProcessType type = fElectromagnetic);
        ~DsG4Scintillation();


    
    public:

        G4bool IsApplicable(const G4ParticleDefinition& aParticleType);

        G4double GetMeanFreePath(const G4Track& aTrack, G4ForceCondition* );

        G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep);

        G4VParticleChange* AtRestDoIt (const G4Track& aTrack,
                                       const G4Step& aStep);

	    void SetTrackSecondariesFirst(const G4bool state);

        G4bool GetTrackSecondariesFirst() const;

        void SetScintillationYieldFactor(const G4double yieldfactor);

        G4double GetScintillationYieldFactor() const;

        void SetScintillationExcitationRatio(const G4double excitationratio);

        G4double GetScintillationExcitationRatio() const;

        G4PhysicsTable* GetFastIntegralTable() const;

        G4PhysicsTable* GetSlowIntegralTable() const;

        G4PhysicsTable* GetReemissionIntegralTable() const;

        void DumpPhysicsTable() const;


    public:
        G4PhysicsTable* getSlowIntegralTable();
        G4PhysicsTable* getFastIntegralTable();
        G4PhysicsTable* getReemissionIntegralTable();


    private:
        void BuildThePhysicsTable();


    protected:
        G4PhysicsTable* theSlowIntegralTable;
        G4PhysicsTable* theFastIntegralTable;
        G4PhysicsTable* theReemissionIntegralTable;

        // Birks constant C1 and C2
        double  birksConstant1;
        double  birksConstant2;

        double  slowerTimeConstant;
        double  slowerRatio;

        double gammaSlowerTime;
        double gammaSlowerRatio;


    private:
        G4bool fTrackSecondariesFirst;
        G4double YieldFactor;
        G4double ExcitationRatio;

        bool fEnableQuenching;
};


////////////////////
// Inline methods
////////////////////

inline 
G4bool DsG4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
{
        if (aParticleType.GetParticleName() == "opticalphoton"){
           return true;
        } else {
           return true;
        }
}

inline 
void DsG4Scintillation::SetTrackSecondariesFirst(const G4bool state) 
{ 
	fTrackSecondariesFirst = state;
}

inline
G4bool DsG4Scintillation::GetTrackSecondariesFirst() const
{
        return fTrackSecondariesFirst;
}

inline
void DsG4Scintillation::SetScintillationYieldFactor(const G4double yieldfactor)
{
        YieldFactor = yieldfactor;
}


inline
G4double DsG4Scintillation::GetScintillationYieldFactor() const
{
        return YieldFactor;
}


inline
void DsG4Scintillation::SetScintillationExcitationRatio(const G4double excitationratio)
{
        ExcitationRatio = excitationratio;
}

inline
G4double DsG4Scintillation::GetScintillationExcitationRatio() const
{
        return ExcitationRatio;
}

inline
G4PhysicsTable* DsG4Scintillation::GetSlowIntegralTable() const
{
        return theSlowIntegralTable;
}

inline
G4PhysicsTable* DsG4Scintillation::GetFastIntegralTable() const
{
        return theFastIntegralTable;
}

inline
G4PhysicsTable* DsG4Scintillation::GetReemissionIntegralTable() const
{
 	return theReemissionIntegralTable;
}

inline
void DsG4Scintillation::DumpPhysicsTable() const
{
        if (theFastIntegralTable) {
           G4int PhysicsTableSize = theFastIntegralTable->entries();
           G4PhysicsOrderedFreeVector *v;

           for (G4int i = 0 ; i < PhysicsTableSize ; i++ )
           {
        	v = (G4PhysicsOrderedFreeVector*)(*theFastIntegralTable)[i];
        	v->DumpValues();
           }
         }

        if (theSlowIntegralTable) {
           G4int PhysicsTableSize = theSlowIntegralTable->entries();
           G4PhysicsOrderedFreeVector *v;

           for (G4int i = 0 ; i < PhysicsTableSize ; i++ )
           {
                v = (G4PhysicsOrderedFreeVector*)(*theSlowIntegralTable)[i];
                v->DumpValues();
           }
         }

        if (theReemissionIntegralTable) {
           G4int PhysicsTableSize = theReemissionIntegralTable->entries();
           G4PhysicsOrderedFreeVector *v;

           for (G4int i = 0 ; i < PhysicsTableSize ; i++ )
           {
                v = (G4PhysicsOrderedFreeVector*)(*theReemissionIntegralTable)[i];
                v->DumpValues();
           }
         }
}

#endif /* DsG4Scintillation_h */




// Author : Miao Yu
// Mail : miaoyu@whu.edu.cn
// Description : Rayleigh scattering of anisotropic molecules
// Date : March, 2022


#ifndef RayleighScattering_h
#define RayleighScattering_h 1

#include "globals.hh"
#include "templates.hh"
#include "Randomize.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4ParticleMomentum.hh"
#include "G4Step.hh"
#include "G4VDiscreteProcess.hh"
#include "G4DynamicParticle.hh"
#include "G4Material.hh"
#include "G4OpticalPhoton.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicsOrderedFreeVector.hh"
#include "G4MaterialPropertiesTable.hh"

class RayleighScattering : public G4VDiscreteProcess
{

public:

    explicit RayleighScattering(const G4String& processName= "RayleighScattering",
                                        G4ProcessType type = fOptical);
    virtual ~RayleighScattering();


public:

    virtual G4bool IsApplicable(const G4ParticleDefinition& aParticleType) override;

    virtual void BuildPhysicsTable(const G4ParticleDefinition& aParticleType) override;

    virtual G4double GetMeanFreePath(const G4Track& aTrack,
                                     G4double, 
                                     G4ForceCondition*) override;
    
    virtual G4VParticleChange* PostStepDoIt(const G4Track& aTrack,
                                            const G4Step&  aStep) override;

    virtual G4PhysicsTable* GetPhysicsTable() const;

    virtual void DumpPhysicsTable() const;

protected:
    G4PhysicsTable* thePhysicsTable;

private:

    RayleighScattering(const RayleighScattering &right) = delete;
    RayleighScattering& operator=(const RayleighScattering &right) = delete;

    G4PhysicsOrderedFreeVector*
    CalculateRayleighMeanFreePaths( const G4Material* material) const;

private:
  G4double rhov;

};



inline
G4bool RayleighScattering::IsApplicable(const G4ParticleDefinition& aParticleType)
{
  return (&aParticleType == G4OpticalPhoton::OpticalPhoton());
}

inline
void RayleighScattering::DumpPhysicsTable() const
{
  G4int PhysicsTableSize = thePhysicsTable->entries();
  G4PhysicsOrderedFreeVector *v;

  for (G4int i = 0; i < PhysicsTableSize; ++i)
  {
    v = (G4PhysicsOrderedFreeVector*)(*thePhysicsTable)[i];
    v->DumpValues();
  }
}

inline G4PhysicsTable* RayleighScattering::GetPhysicsTable() const
{
  return thePhysicsTable;
}

#endif /* G4OpRayleigh_h */

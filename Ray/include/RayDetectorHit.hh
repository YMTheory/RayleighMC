
/*************************************************************************
  @Author: MiaoYu ---> miaoyu@ihep.ac.cn
  @Created Time : Fri Sep 25 15:53:38 2020
  @File Name: RayDetectorHit.hh
 ************************************************************************/

#ifndef RayDetectorHit_h
#define RayDetectorHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "tls.hh"

class RayDetectorHit : public G4VHit
{
    public:
        RayDetectorHit();
        RayDetectorHit(const RayDetectorHit&);
        virtual ~RayDetectorHit();

        // operators
        const RayDetectorHit& operator=(const RayDetectorHit&);
        G4int operator==(const RayDetectorHit&) const;

        inline void* operator new    (size_t);
        inline void operator  delete (void*);

        // methods from base class
        virtual void Draw();
        virtual void Print();

        // Set methods
        void SetTrackID(G4int trackId) { fTrackID = trackId; }
        void SetPolX(G4double polx)    { fPolX = polx; }
        void SetPolY(G4double poly)    { fPolY = poly; }
        void SetPolZ(G4double polz)    { fPolZ = polz; }

        // Get methods
        G4int  GetTrackID() const;
        G4double  GetPolX() const;
        G4double  GetPolY() const;
        G4double  GetPolZ() const;
        G4double  GetMomX() const;
        G4double  GetMomY() const;
        G4double  GetMomZ() const;

    private:
        G4double fTrackID;
        G4double fPolX;
        G4double fPolY;
        G4double fPolZ;
        G4double fMomX;
        G4double fMomY;
        G4double fMomZ;

};


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

typedef G4THitsCollection<RayDetectorHit> RayDetectorHitsCollection;

extern G4ThreadLocal G4Allocator<RayDetectorHit>* RayDetectorHitAllocator;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void* RayDetectorHit::operator new(size_t)
{
  if(!RayDetectorHitAllocator)
      RayDetectorHitAllocator = new G4Allocator<RayDetectorHit>;
  return (void *) RayDetectorHitAllocator->MallocSingle();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void RayDetectorHit::operator delete(void *hit)
{
  RayDetectorHitAllocator->FreeSingle((RayDetectorHit*) hit);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline G4int RayDetectorHit::GetTrackID() const {
    return fTrackID;
}



#endif

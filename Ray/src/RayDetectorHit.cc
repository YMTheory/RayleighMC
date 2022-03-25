/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Fri Sep 25 15:58:10 2020
 @File Name: RayDetectorHit.cc
 ************************************************************************/

#include "RayDetectorHit.hh"

G4ThreadLocal G4Allocator<RayDetectorHit>* RayDetectorHitAllocator = 0;

RayDetectorHit::RayDetectorHit()
    : G4VHit(),
      fTrackID(0.)
{}

RayDetectorHit::~RayDetectorHit()
{;}


RayDetectorHit::RayDetectorHit( const RayDetectorHit& right)
    : G4VHit()
{
    fTrackID = right.fTrackID;
}


const RayDetectorHit& RayDetectorHit::operator=(const RayDetectorHit& right)
{
    fTrackID = right.fTrackID;

    return *this;
}

G4int RayDetectorHit::operator==(const RayDetectorHit& right) const
{
    return (  this == &right ) ? 1 : 0;
}
void RayDetectorHit::Draw()
{

}

void RayDetectorHit::Print()
{
    G4cout << " trackID: " << fTrackID << G4endl;
}






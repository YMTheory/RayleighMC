/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Fri Sep 25 13:34:37 2020
 @File Name: RayPhysicsList.hh
 ************************************************************************/

#ifndef RayPhysicsList_h
#define RayPhysicsList_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4VPhysicsConstructor;
class G4ProductionCuts;

class RayPhysicsList : public G4VModularPhysicsList
{
    public:
        RayPhysicsList();
        virtual ~RayPhysicsList();

    public:
        virtual void SetCuts();

        virtual void ConstructParticle();
        virtual void ConstructProcess();
        void ConstructOpticalProcess();



    private:
        G4VPhysicsConstructor* emPhysicsList;
        G4VPhysicsConstructor* decayPhysicsList;
};

#endif

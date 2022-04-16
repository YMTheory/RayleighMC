/*************************************************************************
 @Author: MiaoYu ---> miaoyu@ihep.ac.cn
 @Created Time : Tue Sep 29 16:07:32 2020
 @File Name: RayAnalysisManager.hh
 ************************************************************************/

#ifndef RayAnalysisManager_h
#define RayAnalysisManager_h 1

#include "globals.hh"

class RayAnalysisManager  {
    
    public:

        //RayAnalysisManager();
        virtual ~RayAnalysisManager();

        void book();

        void finish();
        
        //method to call to create an instance of this class
        static RayAnalysisManager* getInstance();
        
        void SetOutputFileName(G4String);

        void analysePhotonNumber(G4int number);
        void analyseEventID(G4int evtid);
        void analyseInitPhotonNumber(G4int number);
        void analyseInitPosY(G4double posy);
        void analyseInitPolX(G4double polx);
        void analyseInitPolY(G4double poly);
        void analyseInitPolZ(G4double polz);
        void analyseDetPosX(G4double posx);
        void analyseDetPosY(G4double posy);
        void analyseDetPosZ(G4double posz);
        void analyseDetPolX(G4double polx);
        void analyseDetPolY(G4double poly);
        void analyseDetPolZ(G4double polz);
        void analyseDetTime(G4double time);
        void analyseAddNtupleRow();


    private:
        RayAnalysisManager();

        G4String outputFileName;

        static RayAnalysisManager* instance;
        
};

#endif

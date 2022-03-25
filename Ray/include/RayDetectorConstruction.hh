
// $Id: RayDetectorConstruction.hh 69565 2013-05-08 12:35:31Z gcosmo $
//
/// \file RayDetectorConstruction.hh
/// \brief Definition of the RayDetectorConstruction class

#ifndef RayDetectorConstruction_h
#define RayDetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;
class G4Material;

/// Detector construction class to define materials and geometry.

class RayDetectorConstruction : public G4VUserDetectorConstruction
{
    public:
        RayDetectorConstruction();
        virtual ~RayDetectorConstruction();

        virtual G4VPhysicalVolume* Construct();
        virtual void ConstructSDandField();

        void DefineMaterials();
        G4VPhysicalVolume* DefineVolumes();


    private:
        G4LogicalVolume* CellConstruction();
        G4LogicalVolume* DetectorConstruction();

    private:
        G4bool fCheckOverlaps;

        G4Material* air;
        G4Material* lab;
        G4Material* black;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif


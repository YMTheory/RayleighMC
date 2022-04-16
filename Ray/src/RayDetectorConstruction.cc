
// $Id: RayDetectorConstruction.cc 94307 2015-11-11 13:42:46Z gcosmo $
//
/// \file RayDetectorConstruction.cc
/// \brief Implementation of the RayDetectorConstruction class

#include "RayDetectorConstruction.hh"
#include "RayDetectorSD.hh"

#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4SubtractionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4SubtractionSolid.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4GeometryManager.hh"
#include "G4OpticalSurface.hh"
#include "G4SDManager.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


RayDetectorConstruction::RayDetectorConstruction()
    : G4VUserDetectorConstruction(),
    fCheckOverlaps(true), air(NULL), lab(NULL)
{ 

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RayDetectorConstruction::~RayDetectorConstruction()
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



G4VPhysicalVolume* RayDetectorConstruction::Construct()
{   

    DefineMaterials();

    return DefineVolumes();
}

void RayDetectorConstruction::DefineMaterials()
{
    G4String name;
    G4double density;
    G4int compNum;

    G4int nEntries = 2;
    G4double photonEnergy[2] = {1.55*eV, 15.5*eV};
    G4double rindex[2] = {1.0, 1.0};
    G4double RayLength[2] = {10000*m, 10000*m};
    G4double Abs[2] = {1.0, 1.0};

    // Get nist material manager
    // air construction
    G4NistManager* nist = G4NistManager::Instance();
    air = nist->FindOrBuildMaterial("G4_AIR");
    G4MaterialPropertiesTable* air_mpt = new G4MaterialPropertiesTable();
    air_mpt -> AddProperty("RINDEX", photonEnergy, rindex, nEntries);
    air_mpt -> AddProperty("RAYLEIGH", photonEnergy, RayLength, nEntries);
    air -> SetMaterialPropertiesTable(air_mpt);


    // lab construction
    lab = new G4Material(name="lab", density=2.23*g/cm3,compNum=2);
    G4MaterialPropertiesTable* lab_mpt = new G4MaterialPropertiesTable();
    lab_mpt -> AddProperty("RINDEX", photonEnergy, rindex, nEntries);
    G4double RayLength1[2] = {27.*m, 27.*m};
    lab_mpt -> AddProperty("RAYLEIGH", photonEnergy, RayLength1, nEntries);
    G4double rhov = 0.0;
    lab_mpt->AddConstProperty("RHOV", rhov);
    lab->SetMaterialPropertiesTable(lab_mpt);

    // black construction
    black = new G4Material(name="black", density=2.0*g/cm3, compNum=1);
    G4MaterialPropertiesTable* black_mpt = new G4MaterialPropertiesTable();
    black_mpt -> AddProperty("RINDEX", photonEnergy, rindex, nEntries);
    black_mpt -> AddProperty("RAYLEIGH", photonEnergy, RayLength, nEntries);
    black_mpt -> AddProperty("theAbsorption", photonEnergy, Abs, nEntries);
    black -> SetMaterialPropertiesTable(black_mpt);
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4LogicalVolume* RayDetectorConstruction::CellConstruction()
{
    
    //G4Box* solidCell  =
    //    new G4Box("cellSolid", 1*cm, 1*cm, 1*cm);
    G4Sphere* solidCell = 
        new G4Sphere("cellSolid", 0*cm, 17700*mm, 0, 2*pi, 0, pi);

    G4ThreeVector zTrans(0, 0, 0);
    G4LogicalVolume* logicCell =                         
        new G4LogicalVolume(
                solidCell,
                lab,             
                "logicCell");        

    return logicCell;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4LogicalVolume* RayDetectorConstruction::DetectorConstruction()
{
    G4Sphere* solidDet = 
        new G4Sphere("detSolid", 18000*mm, 18010*mm, 0, 2*pi, 0, pi);
        //new G4Sphere("detSolid", 10*cm, 11*cm, 0, 2*pi, 0, pi);

    G4Box* solidHole = 
        new G4Box("holeSolid", 1*mm, 1*mm, 2*cm);

    G4VSolid* solidSub = 
        new G4SubtractionSolid("subSolid", solidDet, solidHole, 0, G4ThreeVector(0, 0, 18005*mm));

    G4LogicalVolume* logicDet = 
        new G4LogicalVolume(solidSub, 
                            black,
                            "logicDet");

    


    return logicDet;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* RayDetectorConstruction::DefineVolumes()
{

    G4double worldLength = 40.*m;


    //////////////////////////////////////////////////////////////////////////////////////////////
    // optical surface definition

    // air - glass
    G4OpticalSurface* OpticalAirlab = new G4OpticalSurface("AirLABface");
    OpticalAirlab -> SetModel(unified);
    OpticalAirlab -> SetType(dielectric_dielectric);
    OpticalAirlab -> SetFinish(ground);
    G4MaterialPropertiesTable* AirLABMPT = new G4MaterialPropertiesTable();
    OpticalAirlab->SetMaterialPropertiesTable(AirLABMPT);


    // air - air
    G4OpticalSurface* OpticalAirAir = new G4OpticalSurface("AirAirface");
    OpticalAirAir -> SetModel(unified);
    OpticalAirAir -> SetType(dielectric_dielectric);
    OpticalAirAir -> SetFinish(ground);
    G4MaterialPropertiesTable* myST1 = new G4MaterialPropertiesTable();
    OpticalAirAir->SetMaterialPropertiesTable(myST1);

    //////////////////////////////////////////////////////////////////////////////////////////////

    // Visualization attributes
    G4VisAttributes* boxVisAtt = new G4VisAttributes(G4Colour(0, 1, 1));
    //boxVisAtt->SetForceSolid ();
    boxVisAtt->SetForceWireframe();
    G4VisAttributes* detVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 1));
    //detVisAtt->SetForceAuxEdgeVisible();
    detVisAtt->SetForceWireframe();

    //////////////////////////////////////////////////////////////////////////////////////////////


    // Geometry Construction Part

    //World
    //
    G4GeometryManager::GetInstance()->SetWorldMaximumExtent(worldLength);
    G4Box* worldS
        = new G4Box("world",                                    //its name
                    worldLength/2,worldLength/2,worldLength/2); //its size
    G4LogicalVolume* worldLV
        = new G4LogicalVolume(
                     worldS,   //its solid
                     //air,      //its material
                     air,
                     "World"); //its name
  
    G4VPhysicalVolume* worldPV
     = new G4PVPlacement(
                     0,               // no rotation
                     G4ThreeVector(), // at (0,0,0)
                     worldLV,         // its logical volume
                     "World",         // its name
                     0,               // its mother  volume
                     false,           // no boolean operations
                     0,               // copy number
                     fCheckOverlaps); // checking overlaps 


    //design of cuvette
    //

    G4LogicalVolume* logicCell = CellConstruction();
    G4VPhysicalVolume* physCell = 
        new G4PVPlacement(0,                       
                G4ThreeVector(0, 0, 0), 
                logicCell,                
                "physCell",              
                worldLV,              
                false,                   
                0,                       
                fCheckOverlaps);  
    logicCell   ->  SetVisAttributes(boxVisAtt);  // set visualization
    

    G4LogicalVolume* logicDet = DetectorConstruction();
    G4VPhysicalVolume* physDet =
        new G4PVPlacement(0,
                          G4ThreeVector(0,0,0),
                          logicDet,
                          "physDet", 
                          worldLV,
                          false,
                          0,
                          fCheckOverlaps);
    logicDet -> SetVisAttributes(detVisAtt);

    return worldPV;
}


void RayDetectorConstruction::ConstructSDandField()
{
    G4cout << " ----> Add Sensitive Detector " << G4endl;
    // Sensitive Detector

    auto detectorSD
        =  new RayDetectorSD("detectorSD", "PmtHitsCollection"); //fNofLayers);
    G4SDManager::GetSDMpointer()->AddNewDetector(detectorSD);
    SetSensitiveDetector("logicDet", detectorSD);

}





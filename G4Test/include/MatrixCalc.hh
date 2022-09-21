// A simple matrix and vector calculator used in the anisotropic rayleigh scattering MC
// Author : Miao Yu
// Email : miaoyu@whu.edu.cn
// Date : March, 2022 

#ifndef MatrixCalc_h
#define MatrixCalc_h 1

#include <random>
using namespace std;

class MatrixCalc{

public:
    MatrixCalc();
    ~MatrixCalc();

public:

    static double* cross(double* vec1, double* vec2);
    static double dot(double* vec1, double* vec2);
    static double mag(double* vec);
    static double* norm(double* vec);

    static double* perpendicular_vector(double* vec);
    static double* matrixMultiplier(double* mat1, double* mat2);

    static double* generateRandomMatrix();
    static double* rotatePolTensor(double rhov);
    static double* rotatePolVector(double* mat, double* vec );

};

#endif
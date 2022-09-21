// A simple matrix and vector calculator used in the anisotropic rayleigh scattering MC
// Author : Miao Yu
// Email : miaoyu@whu.edu.cn
// Date : March, 2022 


#include "MatrixCalc.hh"

#include <iostream>
#include <cmath>

MatrixCalc::MatrixCalc()
{;}

MatrixCalc::~MatrixCalc()
{;}


double* MatrixCalc::cross(double* vec1, double* vec2) {
    double x = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    double y = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    double z = vec1[0]*vec2[1] - vec1[1]*vec2[0];

    double* vec = new double[3];
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
    return vec;

}



double MatrixCalc::dot(double* vec1, double* vec2) {
    double dot =  vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    return dot;
}


double MatrixCalc::mag(double* vec) {
    double mag = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    return mag;
}


double* MatrixCalc::norm(double* vec) {
    double mg = mag(vec);
    double* newvec = new double[3];
    newvec[0] = vec[0]/mg;
    newvec[1] = vec[1]/mg;
    newvec[2] = vec[2]/mg;
    return newvec;
}


double* MatrixCalc::perpendicular_vector(double* vec) {
    double vec1[3] = {1, 0, 0};
    double vec2[3] = {0, 1, 1};

    if (vec[2] == 0 and vec[1] == 0){
        if (vec[0] == 0) {
            return vec;
        } else {
            return cross(vec, vec2);
        }
    } else {
        return cross(vec, vec1);
    }
}


double* MatrixCalc::generateRandomMatrix()
{
    static std::default_random_engine generator;
    std::normal_distribution<double> normal(0.0, 1.0);
    double qr = normal(generator);
    double qi = normal(generator);
    double qj = normal(generator);
    double qk = normal(generator);
    double s = 1./(qr*qr+qi*qi+qj*qj+qk*qk);
    double R00 = 1- 2*s*(qj*qj+qk*qk);
    double R01 = 2*s*(qi*qj-qk*qr);
    double R02 = 2*s*(qi*qk+qj*qr);
    double R10 = 2*s*(qi*qj+qk*qr);
    double R11 = 1 - 2*s*(qi*qi+qk*qk);
    double R12 = 2*s*(qj*qk-qi*qr);
    double R20 = 2*s*(qi*qk-qj*qr);
    double R21 = 2*s*(qj*qk+qi*qr);
    double R22 = 1 - 2*s*(qi*qi+qj*qj);

    double* mat = new double[9];
    mat[0] = R00;
    mat[1] = R01;
    mat[2] = R02;
    mat[3] = R10;
    mat[4] = R11;
    mat[5] = R12;
    mat[6] = R20;
    mat[7] = R21;
    mat[8] = R22;
                     
    return mat;
}

double* MatrixCalc::matrixMultiplier(double* mat1, double* mat2) {
    double* mat = new double[9];
    mat[0] = mat1[0]*mat2[0] + mat1[1]*mat2[3] + mat1[2]*mat2[6];
    mat[1] = mat1[0]*mat2[1] + mat1[1]*mat2[4] + mat1[2]*mat2[7];
    mat[2] = mat1[0]*mat2[2] + mat1[1]*mat2[5] + mat1[2]*mat2[8];
    mat[3] = mat1[3]*mat2[0] + mat1[4]*mat2[3] + mat1[5]*mat2[6];
    mat[4] = mat1[3]*mat2[1] + mat1[4]*mat2[4] + mat1[5]*mat2[7];
    mat[5] = mat1[3]*mat2[2] + mat1[4]*mat2[5] + mat1[5]*mat2[8];
    mat[6] = mat1[6]*mat2[0] + mat1[7]*mat2[3] + mat1[8]*mat2[6];
    mat[7] = mat1[6]*mat2[1] + mat1[7]*mat2[4] + mat1[8]*mat2[7];
    mat[8] = mat1[6]*mat2[2] + mat1[7]*mat2[5] + mat1[8]*mat2[8];

    return mat;

}


double* MatrixCalc::rotatePolTensor(double rhov)
{
    double alpha = 1;
    double beta = (std::sqrt(45*rhov)/3-std::sqrt(3-4*rhov)) / (-std::sqrt(3-4*rhov)-2./3.*std::sqrt(45*rhov));
    double tensor[9] = {alpha, 0, 0, 0, beta, 0, 0, 0, beta};

    double* mat = generateRandomMatrix();
    double* tmp = matrixMultiplier(mat, tensor);
    
    double mat_inv[9] = {mat[0], mat[3], mat[6], mat[1], mat[4], mat[7], mat[2], mat[5], mat[8]};
    double* tensor_new = matrixMultiplier(tmp, mat_inv);

    delete[] mat;
    delete[] tmp;
    return tensor_new;
}




double* MatrixCalc::rotatePolVector(double* mat, double* vec)
{
    double x = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
    double y = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
    double z = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];

    double* newvec = new double[3];
    newvec[0] = x;
    newvec[1] = y;
    newvec[2] = z;
    return newvec;
}














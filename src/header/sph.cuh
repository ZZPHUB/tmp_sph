#ifndef __SPH_H_
#define __SPH_H__

#define PI 3.141592654

#include "data_struct.cuh"
#include <iostream>
#include <fstream>

#include "mesh.cuh"
#include "ns.cuh"
#include "io.cuh"

extern __global__ void PC_predictionD(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int numParticles, float* densitydt, float* Veldt, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int* particleIndex, int* sorted_particle_type);

extern __global__ void PC_correctionD(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int* particleIndex, int numParticles, int* sorted_particle_type);

extern void PC_prediction(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, int* particleHash, int* particleIndex, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int numParticles, int* sorted_particle_type);

extern void PC_correction(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, int* particleHash, int* particleIndex, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int numParticles, int* sorted_particle_type);


#endif
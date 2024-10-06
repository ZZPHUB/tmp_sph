#ifndef __NS_H__
#define __NS_H__

extern void SPH_NS_simpleversion(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, float* densitydt, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash, int timestep, float* dofv, float* rhop_sum, float* w_sum);

extern __global__ void computeBoundary_Delta_acoustic_D(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, int* cellStart, int* cellEnd, float* rhop_sum, float* w_sum, int numParticles, int* particleHash, float* dofv);

extern __global__ void computeGovering_equationD(float* sortedPos, float* sortedVel, float* sortedpressure, float* sorteddensity, int* sorted_particle_type, float* densitydt, float* dofv, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash);

#endif
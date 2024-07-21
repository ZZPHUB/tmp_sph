#ifndef __MESH_H__
#define __MESH_H__

///根据位置计算粒子网格坐标
extern __device__ int3 calcGridPos(float3 p);

///根据网格坐标计算hash值
extern __device__ int calcGridHash(int3 gridPos);

//calculate grid hash value for each particle
extern __global__ void calcHashD(float* pos, int* particleHash, int* particleIndex);

///Round a / b to nearest higher integer value
extern int iDivUp(int a, int b);

///compute grid and thread block size for a given number of elements
extern void computeGridSize(int n, int blockSize, int& numBlocks, int& numThreads);

///calculate hash value
extern void calcHash(float* pos, int* particleHash, int* particleIndex, int numParticles);

///sort particleindex by hash value
extern void sortParticles(int* particleHash, int* particleIndex, int numParticles);

///rearrange particle data into sorted order, and find the start of each cell in the sorted hash array
extern __global__ void reorderD(int* particleHash, int* particleIndex, int* cellStart, int* cellEnd, float* oldPos, float* oldVel, float* olddensity, float* oldpressure, int* oldparticle_type, int* oldparticle_zone, float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sortedparticle_type, int* sortedparticle_zone,int *,int *);

///on cpu rearrange particle data
extern void reorder(float* oldPos, float* oldVel, float* olddensity, float* oldpressure, int* oldparticle_type, int* oldparticle_zone, float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sortedparticle_type, int* sortedparticle_zone, int* particleHash, int* particleIndex, int* dcellStart, int* dCellEnd, int numParticles, SimParams* temps,int *,int *);

//extern __global__ void recoverD(float* d_pos, float* d_vel, float* d_density, float* d_pressure, int* d_particleIndex, int* d_particle_type, int* d_particle_zone, float* d_sortedpos, float* d_sortedvel, float* d_sorteddensity, float* d_sortedpressure, int* d_sorted_particle_type, int* d_sorted_particle_zone, int numParticles);

//extern void recover(float* d_pos, float* d_vel, float* d_density, float* d_pressure, int numParticles, int* d_particleIndex, int* d_particle_type, int* d_particle_zone, float* d_sortedpos, float* d_sortedvel, float* d_sorteddensity, float* d_sortedpressure, int* d_sorted_particle_type, int* d_sorted_particle_zone);

#endif
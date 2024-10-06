#include "sph.cuh"
#include <cuda_fp16.h>

using namespace std;

void SPH_NS_simpleversion(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, float* densitydt, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash, int timestep, float* dofv, float* rhop_sum, float* w_sum)
{
    int numThreads, numBlocks;
    //computeGridSize(numParticles, 128, numBlocks, numThreads); 
    computeGridSize(numParticles, 256, numBlocks, numThreads); 

    computeBoundary_Delta_acoustic_D<<<numBlocks,numThreads>>>(sortedPos, sortedVel, sorteddensity, sortedpressure, sorted_particle_type, cellStart, cellEnd, rhop_sum, w_sum, numParticles, particleHash, dofv);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "SPH_NS launch failed: " << cudaGetErrorString(cudaStatus)<< __LINE__ << endl;
        system("pause");
    }

    computeGovering_equationD<<<numBlocks,numThreads>>>(sortedPos, sortedVel, sortedpressure, sorteddensity, sorted_particle_type, densitydt, dofv, Veldt, cellStart, cellEnd, numParticles, particleHash);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "SPH_NS launch failed: " << cudaGetErrorString(cudaStatus)<< __LINE__ << endl;
        system("pause");
    }
}

__global__ void computeBoundary_Delta_acoustic_D(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sorted_particle_type, int* cellStart, int* cellEnd, float* rhop_sum, float* w_sum, int numParticles, int* particleHash, float* dofv)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numParticles)
    {
        //dofv[index] = 0.0; rhop_sum[index] = 0.0; w_sum[index] = 0.0;//dofv-acoustic damper
        float rhop_sum_tmp = 0.0f;
        float w_sum_tmp = 0.0f;
        float dofv_tmp = 0.0f;
        float3 pos,vel;
        pos.x = sortedPos[3 * index];
        pos.y = sortedPos[3 * index + 1];
        pos.z = sortedPos[3 * index + 2];
        vel.x = sortedVel[3 * index];
        vel.y = sortedVel[3 * index + 1];
        vel.z = sortedVel[3 * index + 2];
        int type_0 = sorted_particle_type[index];
        //float dens_0 = sorteddensity[index];
        //float pres_0 = sortedpressure[index];

        int type_1;
        float dens_1,pres_1;
        float dx,dy,dz;
        float rr,q,w,fr;
        float dvdx;
        __half half_dx;
        __half half_dy;
        __half half_dz;
        __half half_rr;

        for (int z = -3; z <= 3; z++)
        {
            for (int y = -3; y <= 3; y++)
            {

                    int newgridHash = particleHash[index] + z*par.gridxdim*par.gridydim + y*par.gridxdim;
                    if (newgridHash <= par.hash_max-3 && newgridHash >= 3)
                    {
                        //int startIndex = cellStart[newgridHash];
                        #define startIndex (cellStart[newgridHash-3])
                        //if (startIndex == 0xffffffff)	continue;
                        //int endIndex = cellEnd[newgridHash];
                        #define endIndex (cellEnd[newgridHash+3])
                        //  iterate over particles in this cell
                        for (int i = startIndex; i < endIndex; i++)
                        {
                            #undef startIndex
                            #undef endIndex
                            //int cellData = particleHash[i];
                            //if (cellData != newgridHash)  break;
                            if (i != index)	// check not colliding with self
                            {
                                dx = pos.x - sortedPos[3 * i];
                                dy = pos.y - sortedPos[3 * i + 1];
                                dz = pos.z - sortedPos[3 * i + 2];
                                half_dx = __float2half(dx);
                                half_dy = __float2half(dy);
                                half_dz = __float2half(dz);
                                half_rr = (half_dx*half_dx + half_dy*half_dy + half_dz*half_dz)/(__float2half(par.h)*__float2half(par.h));
                                //grid_count++;

                                if (half_rr <= __float2half(4.0f))
                                {
                                    pres_1 = sortedpressure[i];
                                    dens_1 = sorteddensity[i];
                                    type_1 = sorted_particle_type[i];

                                    rr = dx * dx + dy * dy + dz * dz;
                                    dvdx = (vel.x - sortedVel[3 * i])*dx + (vel.y - sortedVel[3 * i + 1])*dy + (vel.z - sortedVel[3 * i +2])*dz;
                                    q = sqrtf(rr)/par.h;
                                    fr = (1.0f - q/2.0f) * (1.0f - q/2.0f) * (1.0f - q/2.0f);
                                    w = fr*(1.0f - q/2.0f);
                                    w *= (2.0f*q + 1)*par.adh;
                                    //fr *= -5.0f*par.adh/(par.h*par.h);
                                    fr *= -5.0f*par.adh/(par.h2);
                                    
                                    if (type_0 != 1 && type_1 == 1)//计算边界所需变量
                                    {
                                        rhop_sum_tmp += (pres_1 - dens_1 * (0.0f * dx + 0.0f * dy + (0.0f - par.gravity) * dz)) * w;
                                        w_sum_tmp += w;
                                    }
                                    else if (type_0 == 1 && type_1 == 1)
                                    {
                                        dofv_tmp -= fr *dvdx * par.particleMass / dens_1;
                                    }
                                }
                            }
                        }
                    }
            }
        }
        dofv[index] = dofv_tmp;
        //if(par_count > 128) printf("the ptc :%d has %d pars and it's grid has %d ptcs!\n",index,par_count,grid_count);
        if (sorted_particle_type[index] != 1)
        {
            //if (fabs(w_sum[index]) > 1.0E-8)
            if (fabs(w_sum_tmp) > 1.0E-8f)
            {
                //sortedpressure[index] = rhop_sum[index] / w_sum[index];
                rhop_sum_tmp = rhop_sum_tmp / w_sum_tmp;
            }
            else
            {
                //sortedpressure[index] = 0;
                rhop_sum_tmp = 0.0f;
            }
            //if (sortedpressure[index] < 0)  sortedpressure[index] = 0;
            if(rhop_sum_tmp < 0.0f) rhop_sum_tmp = 0.0f;
            sortedpressure[index] = rhop_sum_tmp;
            sorteddensity[index] = rhop_sum_tmp /(par.cs2) + par.restDensity;
            //sorteddensity[index] = rhop_sum_tmp /(par.cs*par.cs) + par.restDensity;
        }
    }
}


__global__ void computeGovering_equationD(float* sortedPos, float* sortedVel, float* sortedpressure, float* sorteddensity, int* sorted_particle_type, float* densitydt, float* dofv, float* Veldt, int* cellStart, int* cellEnd, int numParticles, int* particleHash)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numParticles)
    {
        float3 pos, vel;
        pos.x = sortedPos[3 * index];
        pos.y = sortedPos[3 * index + 1];
        pos.z = sortedPos[3 * index + 2];
        vel.x = sortedVel[3 * index];
        vel.y = sortedVel[3 * index + 1];
        vel.z = sortedVel[3 * index + 2];
        float pres_0 = sortedpressure[index];	
        float dens_0 = sorteddensity[index];
        float dofv_0 = dofv[index];
        int type_0 = sorted_particle_type[index];

        float dx,dy,dz;
        float dvdx;
        float pres_1,dens_1,dofv_1;
        //int type_1;

        float rr,fr;
        float q;
        float vtmp;

        float densitydt_temp = 0.0f;
        float3 veldt_temp = make_float3(0.0f, 0.0f, 0.0f);
        __half half_dx;
        __half half_dy;
        __half half_dz;
        __half half_rr;

        for (int z = -3; z <= 3; z++)
        {
            for (int y = -3; y <= 3; y++)
            {
                    int gridHash = particleHash[index] + z*par.gridxdim*par.gridydim + y*par.gridxdim ;

                    if (gridHash <= par.hash_max-3 && gridHash >= 3)
                    {
                        //int startIndex = cellStart[gridHash];
                        #define startIndex cellStart[gridHash-3]
                        //if (startIndex == 0xffffffff)	continue;
                        //int endIndex = cellEnd[gridHash];
                        #define endIndex cellEnd[gridHash+3]
                        //  iterate over particles in this cell
                        for (int i = startIndex; i < endIndex; i++)
                        {
                            #undef startIndex
                            #undef endIndex
                            //int cellData = particleHash[i];
                            //if (cellData != gridHash)  break;
                            if (i != index)	// check not colliding with self
                            {
                                dx = pos.x - sortedPos[3*i];
                                dy = pos.y - sortedPos[3*i+1];
                                dz = pos.z - sortedPos[3*i+2];

                                half_dx = __float2half(dx);
                                half_dy = __float2half(dy);
                                half_dz = __float2half(dz);
                                half_rr = (half_dx*half_dx + half_dy*half_dy + half_dz*half_dz)/(__float2half(par.h)*__float2half(par.h));
                                

                                if (half_rr <= __float2half(4.0f))
                                {
                                    dvdx = (vel.x - sortedVel[3*i])*dx + (vel.y - sortedVel[3*i+1])*dy + (vel.z - sortedVel[3*i+2])*dz;
                                    pres_1 = sortedpressure[i];
                                    dens_1 = sorteddensity[i];

                                    dofv_1 = dofv[i];
                                    //type_1 = sorted_particle_type[i];
                                
                                    rr = dx*dx + dy*dy + dz*dz;
                                    q = sqrtf(rr)/par.h;
                                    //fr = -5.0f * par.adh * (1.0f-q/2.0f) * (1.0f-q/2.0f) * (1.0f-q/2.0f)/(par.h*par.h);
                                    fr = -5.0f * par.adh * (1.0f-q/2.0f) * (1.0f-q/2.0f) * (1.0f-q/2.0f)/(par.h2);
                                    
                                    //densitydt_temp += (dens_0*dvdx*fr + (dens_0-dens_1)*rr*fr*par.delta*par.h*par.cs/(rr+par.eta*par.eta))*par.particleMass/dens_1;
                                    densitydt_temp += (dens_0*dvdx*fr + (dens_0-dens_1)*rr*fr*par.delta_h_cs/(rr+par.eta2))*par.particleMass/dens_1;
                                    vtmp = -(pres_0+pres_1)*par.particleMass*fr/(dens_0*dens_1);

                                    if (type_0 == 1 && sorted_particle_type[i] == 1)
                                    {
                                        vtmp += par.h_cs_rho_m*fr*(dofv_0+dofv_1+par.afa*dvdx/(rr+par.eta2))/(dens_0*dens_1);
                                        //vtmp += par.h*par.cs*par.restDensity*par.particleMass*fr*(dofv_0+dofv_1+par.afa*dvdx/(rr+par.eta*par.eta))/(dens_0*dens_1);
                                    }
                                    veldt_temp.x += vtmp*dx;
                                    veldt_temp.y += vtmp*dy;
                                    veldt_temp.z += vtmp*dz;
                                }
                            }
                        }
                    }
             
            }
        }
        __syncthreads();
        densitydt[index] = densitydt_temp;
        Veldt[3 * index] = veldt_temp.x;
        Veldt[3 * index + 1] = veldt_temp.y;
        Veldt[3 * index + 2] = veldt_temp.z + par.gravity;
    }
}

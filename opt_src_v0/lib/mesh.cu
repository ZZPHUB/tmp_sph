#include "sph.cuh"

using namespace std;

///根据位置计算粒子网格坐标
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    float3 gp;
    gp.x = (p.x - par.gridxmin) / par.gridSize;
    gp.y = (p.y - par.gridymin) / par.gridSize;
    gp.z = (p.z - par.gridzmin) / par.gridSize;
    gridPos.x = (int)floorf(gp.x);
    gridPos.y = (int)floorf(gp.y);
    gridPos.z = (int)floorf(gp.z);
    return gridPos;
}

///根据网格坐标计算hash值
__device__ int calcGridHash(int3 gridPos)
{
    return (gridPos.z * par.gridxdim * par.gridydim) + (gridPos.y * par.gridxdim) + gridPos.x;
}

//calculate grid hash value for each particle
__global__ void calcHashD(float* pos, int* particleHash, int* particleIndex)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < par.numParticles)
    {
        float3 p;
        p.x = pos[3 * index];
        p.y = pos[3 * index + 1];
        p.z = pos[3 * index + 2];

        // get address in grid
        int3 gridPos = calcGridPos(p);
        int gridHash = calcGridHash(gridPos);
        // store grid hash and particle index
        particleHash[index] = gridHash;
        particleIndex[index] = index;
    }
}

///Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

///compute grid and thread block size for a given number of elements
void computeGridSize(int n, int blockSize, int& numBlocks, int& numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}

///calculate hash value
void calcHash(float* pos, int* particleHash, int* particleIndex, int numParticles)
{
    int numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    calcHashD <<< numBlocks, numThreads >>> (pos, particleHash, particleIndex);

    //cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "calcHashD launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }
}

///sort particleindex by hash value
void sortParticles(int* particleHash, int* particleIndex, int numParticles)
{

    thrust::sort_by_key(thrust::device_ptr<int>(particleHash),
        thrust::device_ptr<int>(particleHash + numParticles),
        thrust::device_ptr<int>(particleIndex));

    //cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    //int left = 0;
    //int right = numParticles - 1;
    //sortParticles_simple_quicksortD <<<1, 1 >>> (particleHash, particleIndex, left, right, 0);

}

///rearrange particle data into sorted order, and find the start of each cell in the sorted hash array
__global__ void reorderD(int* particleHash, int* particleIndex, int* cellStart, int* cellEnd, float* oldPos, float* oldVel, float* olddensity, float* oldpressure, int* oldparticle_type, int* oldparticle_zone, float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sortedparticle_type, int* sortedparticle_zone,int *table_l,int *table_r)
{
    __shared__ int sharedHash[257]; // blockSize + 1 elements

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;//第几个粒子，这时已经是按照hash排列的顺序了

    if (index < par.numParticles)
    {
        int hash = particleHash[index];

        // Load hash data into shared memory so that we can look 
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;//block中的hash值分布

        if (index > 0 && threadIdx.x == 0)//除了第一个block的每个block的0号thread
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = particleHash[index - 1];
        }
        __syncthreads();

        // If this particle has a different cell index to the previous particle
        // then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])//hash代表threadidx.x处的，sharedhash[threadidx.x]代表threadidx.x-1处的
        {
            cellStart[hash] = index;

            if (index > 0) cellEnd[sharedHash[threadIdx.x]] = index;
        }
        if (index == par.numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos, vel, density and pressure data
        int sortedIndex = particleIndex[index];
        sortedPos[3 * index] = oldPos[3 * sortedIndex];
        sortedPos[3 * index + 1] = oldPos[3 * sortedIndex + 1];
        sortedPos[3 * index + 2] = oldPos[3 * sortedIndex + 2];
        sortedVel[3 * index] = oldVel[3 * sortedIndex];
        sortedVel[3 * index + 1] = oldVel[3 * sortedIndex + 1];
        sortedVel[3 * index + 2] = oldVel[3 * sortedIndex + 2];
        sorteddensity[index] = olddensity[sortedIndex];
        sortedpressure[index] = oldpressure[sortedIndex];
        sortedparticle_type[index] = oldparticle_type[sortedIndex];
        table_l[index] = table_r[sortedIndex] + (sortedIndex - index);//zzp add
        sortedparticle_zone[index] = oldparticle_zone[sortedIndex];
    }
}

void reorder(float* oldPos, float* oldVel, float* olddensity, float* oldpressure, int* oldparticle_type, int* oldparticle_zone, float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int* sortedparticle_type, int* sortedparticle_zone, int* particleHash, int* particleIndex, int* dcellStart, int* dCellEnd, int numParticles, SimParams* temps,int *table_l,int *table_r)
{
    int numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    // set all cells to empty

    cudaMemset(dcellStart, 0xffffffff, temps->numCells * sizeof(int));

    reorderD <<< numBlocks, numThreads >>> (particleHash, particleIndex, dcellStart, dCellEnd, oldPos, oldVel, olddensity, oldpressure, oldparticle_type, oldparticle_zone, sortedPos, sortedVel, sorteddensity, sortedpressure, sortedparticle_type, sortedparticle_zone,table_l,table_r);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "reorderD launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }
}

/*

__global__ void recoverD(float* d_pos, float* d_vel, float* d_density, float* d_pressure, int* d_particleIndex, int* d_particle_type, int* d_particle_zone, float* d_sortedpos, float* d_sortedvel, float* d_sorteddensity, float* d_sortedpressure, int* d_sorted_particle_type, int* d_sorted_particle_zone, int numParticles)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numParticles)
    {
        int real_index = d_particleIndex[index];
        d_pos[3 * real_index] = d_sortedpos[3 * index];
        d_pos[3 * real_index + 1] = d_sortedpos[3 * index + 1];
        d_pos[3 * real_index + 2] = d_sortedpos[3 * index + 2];
        d_vel[3 * real_index] = d_sortedvel[3 * index];
        d_vel[3 * real_index + 1] = d_sortedvel[3 * index + 1];
        d_vel[3 * real_index + 2] = d_sortedvel[3 * index + 2];
        d_density[real_index] = d_sorteddensity[index];
        d_pressure[real_index] = d_sortedpressure[index];
        d_particle_type[real_index] = d_sorted_particle_type[index];
        d_particle_zone[real_index] = d_sorted_particle_zone[index];
    }
}

void recover(float* d_pos, float* d_vel, float* d_density, float* d_pressure, int numParticles, int* d_particleIndex, int* d_particle_type, int* d_particle_zone, float* d_sortedpos, float* d_sortedvel, float* d_sorteddensity, float* d_sortedpressure, int* d_sorted_particle_type, int* d_sorted_particle_zone)
{
    int numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    recoverD << < numBlocks, numThreads >> > (d_pos, d_vel, d_density, d_pressure, d_particleIndex, d_particle_type, d_particle_zone, d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone, numParticles);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "recoverD launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }
}
*/
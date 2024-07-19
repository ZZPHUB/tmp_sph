#include "sph.cuh"

using namespace std;
using namespace chrono;

__constant__ SimParams par;

int main(void)
{
    ///cpu solver--------------------------------------------------------------------------
    float vlx = 3.221, vly = 1, vlz = 1;//整个计算域大小
    float dx = 0.01;//粒子间距
    float vlx_wl = 1.228, vly_wl = 1, vlz_wl = 0.55;//水域大小
    float rho0 = 1000;//初始密度
    float g = -9.81, cs = 25;
    float dt = 2e-4;//时间步
    float sim_time = 7;//总计算时间
    float out_puttime = 0.005;
    int out_putstep = floor(out_puttime / dt);
    int tot_n = floor(sim_time / dt) + out_putstep + 1;
    const int nb = 470903;//虚粒子数目
    const int np = 1140638;//总粒子数

    int measuring_particle_p1 = 459430 - 1;//注意c++下标起始
    int measuring_particle_p3 = 459434 - 1;
    int measuring_particle_p5 = 460771 - 1;
    int measuring_particle_p7 = 463559 - 1;
    int* measuring_particle = new int[4];
    measuring_particle[0] = measuring_particle_p1;
    measuring_particle[1] = measuring_particle_p3;
    measuring_particle[2] = measuring_particle_p5;
    measuring_particle[3] = measuring_particle_p7;


    //流体变量
    //0代表固壁边界，1代表流体，2代表刚体
    float* h_pos = new float[np * 3];
    float* h_vel = new float[np * 3];
    float* h_ac = new float[np * 3];
    float* rhop = new float[np];
    float* p = new float[np];
    int* particle_type = new int[np];
    int* particle_zone = new int[np];//2D里用来进行粒子位移操作，3D里没使用
    int* h_vp_index = new int[np];//virtual particle index，纯流体求解器不使用

    //导入流体数据===(流体按照3*k、3*k+1、3*k+2以行优先格式存储) 
    ini_fluid(h_pos, h_vel, h_ac, rhop, p, particle_type, particle_zone, np);

    
    //output to screen
    cout << "初始化完成，载入固体流体数据！" << endl;
    cout << "========================================" << endl;
    cout << "流体粒子 " << np << endl;
    //cout << "固体节点 " << nnode << " 个 " << "固体单元 " << nele << " 个" << endl;

    int cnt_frame = 0;
    ofstream ofile;

    ///constant variables set--------------------------------------------------------------
    SimParams* temps = (SimParams*)malloc(sizeof(SimParams));
    temps->timeStep = dt;
    temps->cs = cs;
    temps->numParticles = np;
    temps->boundary_p_num = nb;
    temps->gravity = g;
    temps->particleR = dx;
    temps->h = 1.5 * dx;
    temps->kh = 2.0 * temps->h;
    temps->eta = 0.1 * temps->h;
    temps->delta = 0.2;
    temps->gridSize = temps->kh;
    temps->restDensity = rho0;
    temps->particleMass = pow(temps->particleR, 3) * temps->restDensity;
    temps->adh = 21.0 / (16.0 * PI * temps->h * temps->h * temps->h);
    temps->time = 0.0;
    temps->pre_set_domain_x_min = -0.1;
    temps->pre_set_domain_y_min = -0.1;
    temps->pre_set_domain_z_min = -0.1;
    temps->pre_set_domain_x_max = vlx + 0.1;
    temps->pre_set_domain_y_max = vly + 0.1;
    temps->pre_set_domain_z_max = vlz + 0.1;
    temps->gridxmin = temps->pre_set_domain_x_min;
    temps->gridymin = temps->pre_set_domain_y_min;
    temps->gridzmin = temps->pre_set_domain_z_min;
    temps->gridxmax = temps->pre_set_domain_x_max;
    temps->gridymax = temps->pre_set_domain_y_max;
    temps->gridzmax = temps->pre_set_domain_z_max;
    temps->gridxdim = (int)ceil((temps->gridxmax - temps->gridxmin) / temps->gridSize);
    temps->gridydim = (int)ceil((temps->gridymax - temps->gridymin) / temps->gridSize);
    temps->gridzdim = (int)ceil((temps->gridzmax - temps->gridzmin) / temps->gridSize);
    temps->hash_min = 0;
    temps->hash_max = temps->gridxdim * temps->gridydim * temps->gridzdim + temps->gridxdim * temps->gridydim + temps->gridxdim;
    temps->numCells = temps->gridxdim * temps->gridydim * temps->gridzdim;
    temps->namuta_threshold_1 = 0.4;
    temps->namuta_threshold_2 = 0.9;
    temps->CFL = 1.0;
    temps->Ma = 0.01;
    temps->PST_R = 0.2;
    temps->PST_n = 4;
    temps->afa = 0.05;//人工粘性
    temps->w_deltar = temps->adh * pow((1.0 - temps->particleR / temps->h / 2.0), 4) * (2.0 * (temps->particleR / temps->h) + 1.0);

    //SimParams* par_address = nullptr;
    //cudaGetSymbolAddress((void**)&par_address, &par);
    cudaMemcpyToSymbol(par, temps, sizeof(SimParams));//这个cudamemcpytosymbol报错但是编译可以过
    ///constant variables set--------------------------------------------------------------

    ///gpu solver--------------------------------------------------------------------------
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    for (int i = 0; i < ngpus; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        cout << "Device " << i << " " << devProp.name << " has compute capability " << devProp.major << "." << devProp.minor << endl;
    }
    cudaSetDevice(0);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    cout << "Device 0 is used! " << endl;

    // fluid device memory
    float* d_pos = nullptr;//device memory
    float* d_vel = nullptr;
    float* d_density = nullptr;
    float* d_pressure = nullptr;
    int* d_particlehash = nullptr;
    int* d_particleIndex = nullptr;
    float* d_sortedpos = nullptr;
    float* d_sortedvel = nullptr;
    float* d_sorteddensity = nullptr;
    float* d_sortedpressure = nullptr;
    int* d_cellstart = nullptr;
    int* d_cellend = nullptr;
    float* d_densitydt = nullptr;
    float* d_Veldt = nullptr;
    float* d_rhop_sum = nullptr;
    float* d_w_sum = nullptr;
    int* d_particle_type = nullptr;
    int* d_particle_zone = nullptr;
    int* d_sorted_particle_type = nullptr;
    int* d_sorted_particle_zone = nullptr;
    float* d_namutamin = nullptr;
    float* d_normal_x = nullptr;
    float* d_normal_z = nullptr;
    float* d_sorted_normal_x = nullptr;
    float* d_sorted_normal_z = nullptr;
    float4* d_r_m = nullptr;
    float* d_dofv = nullptr;
    float* d_crhopx = nullptr;
    float* d_crhopz = nullptr;
    int* d_inner_index = nullptr;
    float* d_conc = nullptr;
    float* d_fai = nullptr;
    float* d_sigma = nullptr;
    float* d_shifting_F_x = nullptr;
    float* d_shifting_F_z = nullptr;
    float* d_shifting_Gama_x = nullptr;
    float* d_shifting_Gama_z = nullptr;
    float* d_shifting_C_x = nullptr;
    float* d_shifting_C_z = nullptr;
    int d_np = np;
    float* d_pos_tmp = nullptr;
    float* d_vel_tmp = nullptr;
    float* d_density_tmp = nullptr;

    float* h_densitydt = new float[np];
    float* h_Veldt = new float[3 * np];
    int* h_particle_type = new int[np];
    int* h_particle_zone = new int[np];
    float* h_namutamin = new float[np];
    float* h_normal_x = new float[np];
    float* h_normal_z = new float[np];
    float4* h_r_m = new float4[np];

    int sizebyte_f1 = np * sizeof(float);
    int sizebyte_f2 = np * sizeof(float) * 3;
    int sizebyte_ui = np * sizeof(int);
    int sizebyte_cse = temps->numCells * sizeof(int);//cse means cell start end

    cudaMalloc((void**)&d_pos, sizebyte_f2);
    cudaMalloc((void**)&d_vel, sizebyte_f2);
    cudaMalloc((void**)&d_density, sizebyte_f1);
    cudaMalloc((void**)&d_pressure, sizebyte_f1);
    cudaMalloc((void**)&d_sortedpos, sizebyte_f2);
    cudaMalloc((void**)&d_sortedvel, sizebyte_f2);
    cudaMalloc((void**)&d_sorteddensity, sizebyte_f1);
    cudaMalloc((void**)&d_sortedpressure, sizebyte_f1);
    cudaMalloc((void**)&d_cellstart, sizebyte_cse);
    cudaMalloc((void**)&d_cellend, sizebyte_cse);
    cudaMalloc((void**)&d_particlehash, sizebyte_ui);
    cudaMalloc((void**)&d_particleIndex, sizebyte_ui);
    cudaMalloc((void**)&d_densitydt, sizebyte_f1);
    cudaMalloc((void**)&d_Veldt, sizebyte_f2);
    cudaMalloc((void**)&d_pos_tmp, sizebyte_f2);
    cudaMalloc((void**)&d_vel_tmp, sizebyte_f2);
    cudaMalloc((void**)&d_density_tmp, sizebyte_f1);
    cudaMalloc((void**)&d_rhop_sum, sizebyte_f1);
    cudaMalloc((void**)&d_w_sum, sizebyte_f1);
    cudaMalloc((void**)&d_particle_type, sizebyte_ui);
    cudaMalloc((void**)&d_particle_zone, sizebyte_ui);
    cudaMalloc((void**)&d_sorted_particle_type, sizebyte_ui);
    cudaMalloc((void**)&d_sorted_particle_zone, sizebyte_ui);
    cudaMalloc((void**)&d_namutamin, sizebyte_f1);
    cudaMalloc((void**)&d_normal_x, sizebyte_f1);
    cudaMalloc((void**)&d_normal_z, sizebyte_f1);
    cudaMalloc((void**)&d_sorted_normal_x, sizebyte_f1);
    cudaMalloc((void**)&d_sorted_normal_z, sizebyte_f1);
    cudaMalloc((void**)&d_r_m, sizebyte_f2 * 2);
    cudaMalloc((void**)&d_dofv, sizebyte_f1);
    cudaMalloc((void**)&d_crhopx, sizebyte_f1);
    cudaMalloc((void**)&d_crhopz, sizebyte_f1);
    cudaMalloc((void**)&d_inner_index, sizebyte_ui);
    cudaMalloc((void**)&d_conc, sizebyte_f1);
    cudaMalloc((void**)&d_fai, sizebyte_f1);
    cudaMalloc((void**)&d_sigma, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_F_x, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_F_z, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_C_x, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_C_z, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_Gama_x, sizebyte_f1);
    cudaMalloc((void**)&d_shifting_Gama_z, sizebyte_f1);

    cudaMemcpy(d_pos, h_pos, sizebyte_f2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, sizebyte_f2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_density, rhop, sizebyte_f1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_density_tmp, rhop, sizebyte_f1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pressure, p, sizebyte_f1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_type, particle_type, sizebyte_ui, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_zone, particle_zone, sizebyte_ui, cudaMemcpyHostToDevice);

    int* h_particlehash = new int[np];
    int* h_particleIndex = new int[np];
    float* h_sortedpos = new float[np * 3];
    int* h_cellstart = new int[temps->numCells];
    int* h_cellend = new int[temps->numCells];

    auto start_time = steady_clock::now();

    time_t now;
    char* date_now;
    time_t now_start = time(0);
    char* date_time_start = ctime(&now_start);
    cout << "计算开始时间：" << date_time_start << endl;

    ofile.open("./result/Simulation records.dat", ios::out);
    ofile << "This file records the simulation instants!" << endl;
    ofile << "计算开始时间：" << date_time_start << endl;
    ofile.close();

    //控制多线程控制文件输出
    bool file_write_control_1 = true;
    bool file_write_control_2 = true;
    bool file_write_control_3 = true;
    int branch_i = out_putstep;

    for (int i = 0; i < tot_n; i++)
    {
        //复制内存节约时间
        if (((i + 1) % out_putstep) == 1)
        {
            cudaMemcpy(h_pos, d_pos, sizebyte_f2, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vel, d_vel, sizebyte_f2, cudaMemcpyDeviceToHost);
            //cudaMemcpy(rhop, d_density, sizebyte_f1, cudaMemcpyDeviceToHost);
            cudaMemcpy(p, d_pressure, sizebyte_f1, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_particle_type, d_particle_type, sizebyte_ui, cudaMemcpyDeviceToHost);

            now = time(0);
            date_now = ctime(&now);
            cout << "Time step " << i << ",    Simulation time " << i * dt << "s,   Real time: " << date_now << endl;

            ofile.open("./result/Simulation records.dat", ios::out | ios::app);
            ofile << "Time step " << i << ",    Simulation time " << i * dt << "s,   Real time: " << date_now << endl;
            ofile.close();

            cnt_frame++;

            branch_i = i;

            thread output_t1(output_fluid_file, cnt_frame, (i), dt, np, h_pos, h_vel, rhop, p, h_particle_type, h_particle_zone, h_vp_index, measuring_particle, dx, &file_write_control_1);
            output_t1.detach();

            //thread output_t2(output_measured_data_fluid, (i), dt, measuring_particle, np, p, h_pos, dx, &file_write_control_2);
            //output_t2.detach();

        }

        calcHash(d_pos, d_particlehash, d_particleIndex, d_np);

        sortParticles(d_particlehash, d_particleIndex, d_np);

        reorder(d_pos, d_vel, d_density, d_pressure, d_particle_type, d_particle_zone, d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure,
            d_sorted_particle_type, d_sorted_particle_zone, d_particlehash, d_particleIndex, d_cellstart, d_cellend, d_np, temps);//checked
 
        //renormalized_matrix(d_sortedpos, d_sorteddensity, d_sorted_particle_type, d_sorted_particle_zone, d_cellstart, d_cellend, d_np, d_particlehash, d_namutamin, d_normal_x, d_normal_z, d_r_m);

        //particle_zone_detection(d_sortedpos, d_sorted_particle_type, d_sorted_particle_zone, d_namutamin, d_normal_x, d_normal_z, d_cellstart, d_cellend, d_particlehash, d_np);

        //SPH_NS(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone, d_densitydt, d_Veldt, d_cellstart, d_cellend, d_np, d_particlehash, i, d_r_m, d_dofv, d_crhopx, d_crhopz, d_rhop_sum, d_w_sum);

        SPH_NS_simpleversion(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_densitydt, d_Veldt, d_cellstart, d_cellend, d_np, d_particlehash, i, d_dofv, d_rhop_sum, d_w_sum);

        PC_prediction(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_densitydt, d_Veldt, d_particlehash, d_particleIndex, d_pos_tmp, d_vel_tmp, d_density_tmp, d_np, d_sorted_particle_type);

        recover(d_pos, d_vel, d_density, d_pressure, d_np, d_particleIndex, d_particle_type, d_particle_zone, d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone);

        calcHash(d_pos, d_particlehash, d_particleIndex, d_np);

        sortParticles(d_particlehash, d_particleIndex, d_np);

        reorder(d_pos, d_vel, d_density, d_pressure, d_particle_type, d_particle_zone, d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone, d_particlehash, d_particleIndex, d_cellstart, d_cellend, d_np, temps);

        //renormalized_matrix(d_sortedpos, d_sorteddensity, d_sorted_particle_type, d_sorted_particle_zone, d_cellstart, d_cellend, d_np, d_particlehash, d_namutamin, d_normal_x, d_normal_z, d_r_m);

        //particle_zone_detection(d_sortedpos, d_sorted_particle_type, d_sorted_particle_zone, d_namutamin, d_normal_x, d_normal_z, d_cellstart, d_cellend, d_particlehash, d_np);

        //SPH_NS(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone, d_densitydt, d_Veldt, d_cellstart, d_cellend, d_np, d_particlehash, i, d_r_m, d_dofv, d_crhopx, d_crhopz, d_rhop_sum, d_w_sum);

        SPH_NS_simpleversion(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_densitydt, d_Veldt, d_cellstart, d_cellend, d_np, d_particlehash, i, d_dofv, d_rhop_sum, d_w_sum);

        PC_correction(d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_densitydt, d_Veldt, d_particlehash, d_particleIndex, d_pos_tmp, d_vel_tmp, d_density_tmp, d_np, d_sorted_particle_type);

        //concentration(d_sortedpos, d_sorted_particle_type, d_sorted_particle_zone, d_sorteddensity, d_inner_index, d_conc, d_cellstart, d_cellend, d_np, d_particlehash, i);

        //particle_shifting(d_sortedpos, d_sorteddensity, d_sorted_particle_type, d_sorted_particle_zone, d_cellstart, d_cellend, d_np, d_particlehash, d_fai, d_sigma, d_normal_x, d_normal_z, d_shifting_F_x, d_shifting_F_z, d_shifting_Gama_x, d_shifting_Gama_z, d_shifting_C_x, d_shifting_C_z);

        if ((i % 20) == 0)
        {
            //concentration(d_sortedpos, d_sorted_particle_type, d_sorted_particle_zone, d_sorteddensity, d_inner_index, d_conc, d_cellstart, d_cellend, d_np, d_particlehash, i);

            //particle_shifting(d_sortedpos, d_sorteddensity, d_sorted_particle_type, d_sorted_particle_zone, d_cellstart, d_cellend, d_np, d_particlehash, d_fai, d_sigma, d_normal_x, d_normal_z, d_shifting_F_x, d_shifting_F_z, d_shifting_Gama_x, d_shifting_Gama_z, d_shifting_C_x, d_shifting_C_z);

            //density_filter(d_sortedpos, d_sorteddensity, d_cellstart, d_cellend, d_np, d_rhop_sum, d_w_sum, d_particlehash);
        }

        recover(d_pos, d_vel, d_density, d_pressure, d_np, d_particleIndex, d_particle_type, d_particle_zone, d_sortedpos, d_sortedvel, d_sorteddensity, d_sortedpressure, d_sorted_particle_type, d_sorted_particle_zone);

        //SPH_SPIM_FSI_f2s(n_edge, nnode, edge_FSI_force, F_fsi, cur_node_coord, sw_ele_NodeIndex, boundary_edge_cell, s_ele_NodeIndex, d_pos, d_pressure, boundary_node_index);

        //SPIM_solver(nnode, ngp, num_vp,
        //    u_solid, v_solid, a_solid, cur_node_coord, ini_node_coord, constrained_index,
        //    gp_F, gp_spnode, gp_SFDx, gp_SFDy,
        //    gp_BMNL, gp_BML, gp_B, nrows_BMNL,
        //    gp_E,
        //    gp_RST_S,
        //    gp_RST_sigma,
        //    fint, gp_weight, ele_vol, s_ele_NodeIndex, f_ext, 
        //    node_sigma,
        //    F_fsi, mass_inv,
        //    d_pos, d_vel, FSI_s2f, FSI_s2f_sf);

        if (i == (branch_i + out_putstep - 1))
        {
            while (file_write_control_1)
            {

            }
        }
    }

    //cudaDeviceSynchronize();

    time_t now_end = time(0);
    char* date_time_end = ctime(&now_end);
    cout << "计算结束时间：" << date_time_end << endl;
    ofile.open("./result/Simulation records.dat", ios::out | ios::app);
    ofile << "计算结束时间：" << date_time_end << endl;
    ofile.close();

    auto end_time = steady_clock::now();
    float time_cost = duration<float, milli>(end_time - start_time).count();
    cout << "Time for CUDA SPH is " << time_cost / 1000.0 / 60.0 << "minutes" << endl;

    ///gpu solver--------------------------------------------------------------------------

    ///free gpu memory---------------------------------------------------------------------
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_density);
    cudaFree(d_pressure);
    cudaFree(d_sortedpos);
    cudaFree(d_sortedvel);
    cudaFree(d_sorteddensity);
    cudaFree(d_sortedpressure);
    cudaFree(d_cellstart);
    cudaFree(d_cellend);
    cudaFree(d_particlehash);
    cudaFree(d_particleIndex);
    cudaFree(d_densitydt);
    cudaFree(d_Veldt);
    cudaFree(d_pos_tmp);
    cudaFree(d_vel_tmp);
    cudaFree(d_density_tmp);
    cudaFree(d_rhop_sum);
    cudaFree(d_w_sum);
    cudaFree(d_particle_type);
    cudaFree(d_particle_zone);
    cudaFree(d_sorted_particle_type);
    cudaFree(d_sorted_particle_zone);
    cudaFree(d_namutamin);
    cudaFree(d_normal_x);
    cudaFree(d_normal_z);
    cudaFree(d_r_m);
    cudaFree(d_dofv);
    cudaFree(d_crhopx);
    cudaFree(d_crhopz);
    cudaFree(d_inner_index);
    cudaFree(d_conc);
    cudaFree(d_fai);
    cudaFree(d_sigma);
    cudaFree(d_shifting_F_x);
    cudaFree(d_shifting_F_z);
    cudaFree(d_shifting_C_x);
    cudaFree(d_shifting_C_z);
    cudaFree(d_shifting_Gama_x);
    cudaFree(d_shifting_Gama_z);
    ///free gpu memory
    
    return 0;
}


///预测校正积分
__global__ void PC_predictionD(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, int numParticles, float* densitydt, float* Veldt, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int* particleIndex, int* sorted_particle_type)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numParticles)
    {
        int type = sorted_particle_type[index];

        if (type == 1)
        {
            int real_index = particleIndex[index];

            //判断粒子是否跑出计算域
            if (sortedPos[3 * index] > 3.23 || sortedPos[3 * index] < 0 || sortedPos[3 * index + 1] > 1 || sortedPos[3 * index + 1] < 0 || sortedPos[3 * index + 2] > 1 || sortedPos[3 * index + 2] < 0)
            {
                sortedPos[3 * index] = 0;
                sortedPos[3 * index + 1] = 0.5;
                sortedPos[3 * index + 2] = -0.08;
                sortedVel[3 * index] = 0;
                sortedVel[3 * index + 1] = 0;
                sortedVel[3 * index + 2] = 0;
                sortedpressure[index] = 0;
                sorteddensity[index] = par.restDensity;
            }
            else
            {
                Pos_tmp[3 * real_index] = sortedPos[3 * index];
                Pos_tmp[3 * real_index + 1] = sortedPos[3 * index + 1];
                Pos_tmp[3 * real_index + 2] = sortedPos[3 * index + 2];
                Vel_tmp[3 * real_index] = sortedVel[3 * index];
                Vel_tmp[3 * real_index + 1] = sortedVel[3 * index + 1];
                Vel_tmp[3 * real_index + 2] = sortedVel[3 * index + 2];
                density_tmp[real_index] = sorteddensity[index];

                sorteddensity[index] = density_tmp[real_index] + densitydt[index] * par.timeStep / 2.0;//pc scheme分两步推进故除2
                if (sorteddensity[index] < par.restDensity) sorteddensity[index] = par.restDensity;
                sortedPos[3 * index] = Pos_tmp[3 * real_index] + sortedVel[3 * index] * par.timeStep / 2.0;
                sortedPos[3 * index + 1] = Pos_tmp[3 * real_index + 1] + sortedVel[3 * index + 1] * par.timeStep / 2.0;
                sortedPos[3 * index + 2] = Pos_tmp[3 * real_index + 2] + sortedVel[3 * index + 2] * par.timeStep / 2.0;
                sortedVel[3 * index] = Vel_tmp[3 * real_index] + Veldt[3 * index] * par.timeStep / 2.0;
                sortedVel[3 * index + 1] = Vel_tmp[3 * real_index + 1] + Veldt[3 * index + 1] * par.timeStep / 2.0;
                sortedVel[3 * index + 2] = Vel_tmp[3 * real_index + 2] + Veldt[3 * index + 2] * par.timeStep / 2.0;

                sortedpressure[index] = pow(par.cs, 2.0) * (sorteddensity[index] - par.restDensity);
            }
        }
    }
}

__global__ void PC_correctionD(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int* particleIndex, int numParticles, int* sorted_particle_type)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < numParticles)
    {
        int type = sorted_particle_type[index];

        if (type == 1)
        {
            int real_index = particleIndex[index];

            if (sortedPos[3 * index] > 3.23 || sortedPos[3 * index] < 0 || sortedPos[3 * index + 1] > 1 || sortedPos[3 * index + 1] < 0 || sortedPos[3 * index + 2] > 1 || sortedPos[3 * index + 2] < 0)
            {
                sortedPos[3 * index] = 0;
                sortedPos[3 * index + 1] = 0.5;
                sortedPos[3 * index + 2] = -0.08;
                sortedVel[3 * index] = 0;
                sortedVel[3 * index + 1] = 0;
                sortedVel[3 * index + 2] = 0;
                sortedpressure[index] = 0;
                sorteddensity[index] = par.restDensity;
            }
            else
            {
                sorteddensity[index] = density_tmp[real_index] + densitydt[index] * par.timeStep / 2.0;//pc scheme分两步推进故除2
                if (sorteddensity[index] < par.restDensity) sorteddensity[index] = par.restDensity;
                sortedPos[3 * index] = Pos_tmp[3 * real_index] + sortedVel[3 * index] * par.timeStep / 2.0;
                sortedPos[3 * index + 1] = Pos_tmp[3 * real_index + 1] + sortedVel[3 * index + 1] * par.timeStep / 2.0;
                sortedPos[3 * index + 2] = Pos_tmp[3 * real_index + 2] + sortedVel[3 * index + 2] * par.timeStep / 2.0;
                sortedVel[3 * index] = Vel_tmp[3 * real_index] + Veldt[3 * index] * par.timeStep / 2.0;
                sortedVel[3 * index + 1] = Vel_tmp[3 * real_index + 1] + Veldt[3 * index + 1] * par.timeStep / 2.0;
                sortedVel[3 * index + 2] = Vel_tmp[3 * real_index + 2] + Veldt[3 * index + 2] * par.timeStep / 2.0;

                sorteddensity[index] = 2 * sorteddensity[index] - density_tmp[real_index];
                if (sorteddensity[index] < par.restDensity) sorteddensity[index] = par.restDensity;
                sortedPos[3 * index] = 2 * sortedPos[3 * index] - Pos_tmp[3 * real_index];
                sortedPos[3 * index + 1] = 2 * sortedPos[3 * index + 1] - Pos_tmp[3 * real_index + 1];
                sortedPos[3 * index + 2] = 2 * sortedPos[3 * index + 2] - Pos_tmp[3 * real_index + 2];
                sortedVel[3 * index] = 2 * sortedVel[3 * index] - Vel_tmp[3 * real_index];
                sortedVel[3 * index + 1] = 2 * sortedVel[3 * index + 1] - Vel_tmp[3 * real_index + 1];
                sortedVel[3 * index + 2] = 2 * sortedVel[3 * index + 2] - Vel_tmp[3 * real_index + 2];

                sortedpressure[index] = pow(par.cs, 2.0) * (sorteddensity[index] - par.restDensity);
            }
        }
    }
}

void PC_prediction(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, int* particleHash, int* particleIndex, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int numParticles, int* sorted_particle_type)
{
    int numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    PC_predictionD << < numBlocks, numThreads >> > (sortedPos, sortedVel, sorteddensity, sortedpressure, numParticles, densitydt, Veldt, Pos_tmp, Vel_tmp, density_tmp, particleIndex, sorted_particle_type);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "PC_predictionD launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }

}

void PC_correction(float* sortedPos, float* sortedVel, float* sorteddensity, float* sortedpressure, float* densitydt, float* Veldt, int* particleHash, int* particleIndex, float* Pos_tmp, float* Vel_tmp, float* density_tmp, int numParticles, int* sorted_particle_type)
{
    int numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    PC_correctionD << < numBlocks, numThreads >> > (sortedPos, sortedVel, sorteddensity, sortedpressure, densitydt, Veldt, Pos_tmp, Vel_tmp, density_tmp, particleIndex, numParticles, sorted_particle_type);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cout << "PC_correctionD launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        system("pause");
    }
}
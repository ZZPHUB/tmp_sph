#include "sph.cuh"
#include "io.cuh"

using namespace std;

void ini_fluid(float* h_pos, float* h_vel, float* h_ac, float* rhop, float* p, int* particle_type, int* particle_zone, int np)
{
    ifstream infile;

    infile.open("h_pos.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np * 3; i++)
    {
        infile >> h_pos[i];
    }
    infile.close();


    infile.open("h_vel.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np * 3; i++)
    {
        infile >> h_vel[i];
    }
    infile.close();


    infile.open("h_ac.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np * 3; i++)
    {
        infile >> h_ac[i];
    }
    infile.close();

    infile.open("rhop.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np; i++)
    {
        infile >> rhop[i];
    }
    infile.close();

    infile.open("p.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np; i++)
    {
        infile >> p[i];
    }
    infile.close();

    infile.open("particle_type.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np; i++)
    {
        infile >> particle_type[i];
    }
    infile.close();

    infile.open("particle_zone.dat", ios::in);
    if (!infile.is_open())
    {
        cerr << "open file error!" << endl;
        exit(1);
    }
    for (int i = 0; i < np; i++)
    {
        infile >> particle_zone[i];
    }
    infile.close();

}

void thread_loop(int *t_timestep,int t_np,float t_dt,float t_dx,float *t_pos,float *t_vel,float *t_p,int *t_table,int *t_type,int *t_node,std::atomic<int> *t_flag)
{
    int expect_1 = 1;
    int expect_2 = 2;
    int &ref_1 = expect_1;
    int &ref_2 = expect_2;

    while (true)
    {
       // std::cout << "i am in " << *t_timestep << std::endl;
        if(t_flag->compare_exchange_strong(ref_1,2))
        {
            /*out put file*/
            output_fluid_file(*t_timestep,t_np,t_dt,t_dx,t_pos,t_vel,t_p,t_table,t_type,t_node);
            t_flag->compare_exchange_strong(ref_2,0);
        }
        ref_1 = 1;
        ref_2 = 2;

    }
       
}

float trans(float a)
{
	unsigned int aa = *(unsigned int*)(&a);
	unsigned char b[4];
	b[0] = (char)((aa&0xff000000)>>24);
	b[1] = (char)((aa&0x00ff0000)>>16);
	b[2] = (char)((aa&0x0000ff00)>>8);
	b[3] = (char)((aa&0x000000ff)>>0);
	float c = *(float *)b;
	return c;

}


void output_fluid_file(int i,int np,float dt,float dx, float* h_pos, float* h_vel, float* p,int *table,int *h_particle_type, int* measuring_node)
{
    ofstream ofile;
    /*
    ofile.precision(6);
    ofile.setf(ios::showpoint);
    ofile.setf(ios::scientific, ios::floatfield);
    ofile.setf(ios::left, ios::adjustfield);
    */

    string file_name = "./result/fluid_t=" + to_string(i) + ".vtk";
    ofile.open(file_name, ios::out);
    if (!ofile)
    {
        cerr << "open file fail in the main program!" << endl;
        exit(1);
    }
    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "sph data" << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << np << " " << "float" << std::endl;
    float *h_pos_tmp  =new float[np*3];
    float *h_vel_tmp  =new float[np*3];
    float *h_p_tmp  =new float[np];
    int *h_particle_type_tmp = new int[np];
    for(int i=0;i<np;i++)
    {
        h_pos_tmp[(i+table[i])*3] = trans(h_pos[i*3]);
        h_pos_tmp[(i+table[i])*3+1] = trans(h_pos[i*3+1]);
        h_pos_tmp[(i+table[i])*3+2] = trans(h_pos[i*3+2]);
        h_vel_tmp[(i+table[i])*3] = trans(h_vel[i*3]);
        h_vel_tmp[(i+table[i])*3+1] =trans(h_vel[i*3+1]);
        h_vel_tmp[(i+table[i])*3+2] = trans(h_vel[i*3+2]);
        h_p_tmp[i+table[i]] = trans(p[i]);
        h_particle_type_tmp[i+table[i]] = trans(h_particle_type[i]);
    }

    for(int i=0;i<np;i++)
    {
        
        //ofile << h_pos_tmp[i*3] << " " << h_pos_tmp[i*3+1] << " " << h_pos_tmp[i*3+2] <<endl;
    	//ofile.write((char *)(&h_pos_tmp[3*i]),3*sizeof(float));
	    ofile.write((char *)(&h_pos_tmp[3*i]),sizeof(float));
	    ofile.write((char *)(&h_pos_tmp[3*i+1]),sizeof(float));
	    ofile.write((char *)(&h_pos_tmp[3*i+2]),sizeof(float));
    }

    ofile << "POINT_DATA" << " " << np << std::endl;

    ofile << "SCALARS "<< "p float 1" << std::endl;
    ofile << "LOOKUP_TABLE DEFAULT" << std::endl;
    for(int i=0;i<np;i++)
    {
        //ofile << h_p_tmp[i] << endl;
    	ofile.write((char *)(&h_p_tmp[i]),sizeof(float));
    }

    ofile << "VECTORS "<< "velocity float" << std::endl;
    for(int i=0;i<np;i++)
    {
        //ofile << h_vel_tmp[i*3] << " " << h_vel_tmp[i*3+1] << " " << h_vel_tmp[i*3+2] << endl;
    	ofile.write((char *)(&h_vel_tmp[3*i]),3*sizeof(float));
    }

    ofile.close();

    float h2 = 2.228, h4 = 0.582;
    float maxh_h2 = 0, maxh_h4 = 0;

    for (int k = 0; k < np; k++)
    {
        if (h_particle_type_tmp[k] == 1)
        {
            if (h_pos_tmp[3 * k] > (h4 - 1.5 * dx) && h_pos_tmp[3 * k] <(h4 + 1.5 * dx) && h_pos_tmp[3 * k + 1] > (0.5 - 1.5 * dx) && h_pos_tmp[3 * k + 1] < (0.5 + 1.5 * dx))
            {
                maxh_h4 = max(maxh_h4, h_pos_tmp[3 * k + 2]);
            }
            if (h_pos_tmp[3 * k] > (h2 - 1.5 * dx) && h_pos_tmp[3 * k] <(h2 + 1.5 * dx) && h_pos_tmp[3 * k + 1] > (0.5 - 1.5 * dx) && h_pos_tmp[3 * k + 1] < (0.5 + 1.5 * dx))
            {
                if (h_pos_tmp[3 * k + 2] < 0.4)
                {
                    maxh_h2 = max(maxh_h2, h_pos_tmp[3 * k + 2]);
                }
            }
        }
    }

    if (i == 0)
    {
        ofile.open("./result/measuring_fluid_data.dat", ios::out);
        if (!ofile)
        {
            cerr << "open measuring_data file fail in the cuda subroutine!" << endl;
            exit(1);
        }

        ofile << i * dt << "\t" << h_p_tmp[measuring_node[0]] << "\t" << h_p_tmp[measuring_node[1]] << "\t" << h_p_tmp[measuring_node[2]] << "\t" << h_p_tmp[measuring_node[3]] << "\t" << maxh_h2 << "\t" << maxh_h4 << endl;
        ofile.close();
    }
    else
    {
        ofile.open("./result/measuring_fluid_data.dat", ios::out | ios::app);
        if (!ofile)
        {
            cerr << "open measuring_data file fail in the cuda subroutine!" << endl;
            exit(1);
        }
        ofile << i * dt << "\t" << h_p_tmp[measuring_node[0]] << "\t" << h_p_tmp[measuring_node[1]] << "\t" << h_p_tmp[measuring_node[2]] << "\t" << h_p_tmp[measuring_node[3]] << "\t" << maxh_h2 << "\t" << maxh_h4 << endl;
        ofile.close();
    }
    delete[] h_pos_tmp;
    delete[] h_vel_tmp;
    delete[] h_p_tmp;

}

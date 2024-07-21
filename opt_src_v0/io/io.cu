#include "sph.cuh"

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


void output_fluid_file(int cnt_frame, int i, float dt, int np, float* h_pos, float* h_vel, float* rhop, float* p, int* h_particle_type, int* h_particle_zone, int* h_vp_index, int* measuring_node, float dx, bool* file_write_control_1,int *table)
{
    *file_write_control_1 = true;

    ofstream ofile;
    ofile.precision(6);
    ofile.setf(ios::showpoint);
    ofile.setf(ios::scientific, ios::floatfield);
    ofile.setf(ios::left, ios::adjustfield);

    string file_name = "./result/fluid_t=" + to_string(i) + "vtk";
    ofile.open(file_name, ios::out);
    if (!ofile)
    {
        cerr << "open file fail in the main program!" << endl;
        exit(1);
    }
    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "sph data" << std::endl;
    ofile << "ASCII" << std::endl;
    ofile << "DATASET UNSTRUCTURED_GRID" << std::endl;
    ofile << "POINTS " << np << " " << "double" << std::endl;
    float *h_pos_tmp  =new float[np*3];
    float *h_vel_tmp  =new float[np*3];
    float *h_p_tmp  =new float[np];
    for(int i=0;i<np;i++)
    {
        h_pos_tmp[(i+table[i])*3] = h_pos[i*3];
        h_pos_tmp[(i+table[i])*3+1] = h_pos[i*3+1];
        h_pos_tmp[(i+table[i])*3+2] = h_pos[i*3+2];
        h_vel_tmp[(i+table[i])*3] = h_vel[i*3];
        h_vel_tmp[(i+table[i])*3+1] = h_vel[i*3+1];
        h_vel_tmp[(i+table[i])*3+2] = h_vel[i*3+2];
        h_p_tmp[i+table[i]] = p[i];
    }

    for(int i=0;i<np;i++)
    {
        
        ofile << h_pos_tmp[i*3] << " " << h_pos_tmp[i*3+1] << " " << h_pos_tmp[i*3+2] <<endl;
    }

    ofile << "POINT_DATA" << " " << np << std::endl;

    ofile << "SCALARS "<< "p double 1" << std::endl;
    ofile << "LOOKUP_TABLE DEFAULT" << std::endl;
    for(int i=0;i<np;i++)
    {
        ofile << h_p_tmp[i] << endl;
    }

    ofile << "VECTORS "<< "velocity double" << std::endl;
    for(int i=0;i<np;i++)
    {
        ofile << h_vel_tmp[i*3] << " " << h_vel_tmp[i*3+1] << " " << h_vel_tmp[i*3+2] << endl;
    }

    ofile.close();

    float h2 = 2.228, h4 = 0.582;
    float maxh_h2 = 0, maxh_h4 = 0;

    for (int k = 0; k < np; k++)
    {
        if (h_particle_type[k] == 1)
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

    *file_write_control_1 = false;
}
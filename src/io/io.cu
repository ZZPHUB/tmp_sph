#include "sph.cuh"
#include "/home/zzp/workspace/temp/sphio/src/spio.hpp"

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


    //infile.open("pm.dat", ios::in);
    //if (!infile.is_open())
    //{
    //    cerr << "open file error!" << endl;
    //    exit(1);
    //}
    //for (int i = 0; i < np; i++)
    //{
    //    infile >> pm[i];
    //}
    //infile.close();


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

    /*
    double *x = new double[np];
    double *y = new double[np];
    double *z = new double[np];
    double *vx = new double[np];
    double *vy = new double[np];
    double *vz = new double[np];
    //double *p = new double[np];
    //double *rho = new double[np];

    for(int i=0;i<np;i++)
    {
        x[i] = h_pos[i*3];
        y[i] = h_pos[i*3+1];
        z[i] = h_pos[i*3+2];
        vx[i] = h_vel[i*3];
        vy[i] = h_vel[i*3+1];
        vz[i] = h_vel[i*3+2];
        //p = 
    }
    sphvtk ok(x,y,z,vx,vy,vz,p,rhop);
    ok.set_num(np);
    ok.set_path("./");
    ok.writevtk("ok.vtk");
    */


    //infile.open("h_vp_index.dat", ios::in);
    //if (!infile.is_open())
    //{
    //    cerr << "open file error!" << endl;
    //    exit(1);
    //}
    //for (int i = 0; i < np; i++)
    //{
    //    infile >> h_vp_index[i];
    //}
    //infile.close();
}


void output_fluid_file(int cnt_frame, int i, float dt, int np, float* h_pos, float* h_vel, float* rhop, float* p, int* h_particle_type, int* h_particle_zone, int* h_vp_index, int* measuring_node, float dx, bool* file_write_control_1)
{
    *file_write_control_1 = true;

    ofstream ofile;
    ofile.precision(6);
    ofile.setf(ios::showpoint);
    ofile.setf(ios::scientific, ios::floatfield);
    ofile.setf(ios::left, ios::adjustfield);

    string file_name = "./result/fluid_t=" + to_string(i * dt) + "s.dat";
    ofile.open(file_name, ios::out);
    if (!ofile)
    {
        cerr << "open file fail in the main program!" << endl;
        exit(1);
    }
    ofile << "TITLE = FE Data" << endl;
    ofile << "VARIABLES = \"x\", \"y\", \"z\", \"vx\", \"vy\", \"vz\", \"p\"" << endl;

    //ofile << "ZONE T=\"Wall 1 STP:" << cnt_frame << "\", STRANDID = 1, SOLUTIONTIME=" << i * dt << "F = Point" << endl;
    //for (int k = 0; k < np; k++)
    //{
    //    if (h_particle_type[k] == 0)
    //    {
    //        ofile << h_pos[3 * k] << "\t" << h_pos[3 * k + 1] << "\t" << h_pos[3 * k + 2] << "\t" << h_vel[3 * k] << "\t" << h_vel[3 * k + 1] << "\t" << h_vel[3 * k + 2] << "\t" << p[k] << endl;

    //    }
    //}

    ofile << "ZONE T=\"Rigidbody 1 STP:" << cnt_frame << "\", STRANDID = 2, SOLUTIONTIME=" << i * dt << "F = Point" << endl;
    for (int k = 0; k < np; k++)
    {
        if (h_particle_type[k] == 2)
        {
            ofile << h_pos[3 * k] << "\t" << h_pos[3 * k + 1] << "\t" << h_pos[3 * k + 2] << "\t" << h_vel[3 * k] << "\t" << h_vel[3 * k + 1] << "\t" << h_vel[3 * k + 2] << "\t" << p[k] << endl;

        }
    }

    ofile << "ZONE T=\"Fluid 1 STP:" << cnt_frame << "\", STRANDID = 3, SOLUTIONTIME=" << i * dt << "F = Point" << endl;

    for (int k = 0; k < np; k++)
    {
        if (h_particle_type[k] == 1)
        {
            ofile << h_pos[3 * k] << "\t" << h_pos[3 * k + 1] << "\t" << h_pos[3 * k + 2] << "\t" << h_vel[3 * k] << "\t" << h_vel[3 * k + 1] << "\t" << h_vel[3 * k + 2] << "\t" << p[k] << endl;

        }
    }

    ofile.close();


    float h2 = 2.228, h4 = 0.582;
    float maxh_h2 = 0, maxh_h4 = 0;

    for (int k = 0; k < np; k++)
    {
        if (h_particle_type[k] == 1)
        {
            if (h_pos[3 * k] > (h4 - 1.5 * dx) && h_pos[3 * k] <(h4 + 1.5 * dx) && h_pos[3 * k + 1] > (0.5 - 1.5 * dx) && h_pos[3 * k + 1] < (0.5 + 1.5 * dx))
            {
                maxh_h4 = max(maxh_h4, h_pos[3 * k + 2]);
            }
            if (h_pos[3 * k] > (h2 - 1.5 * dx) && h_pos[3 * k] <(h2 + 1.5 * dx) && h_pos[3 * k + 1] > (0.5 - 1.5 * dx) && h_pos[3 * k + 1] < (0.5 + 1.5 * dx))
            {
                if (h_pos[3 * k + 2] < 0.4)
                {
                    maxh_h2 = max(maxh_h2, h_pos[3 * k + 2]);
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

        ofile << i * dt << "\t" << p[measuring_node[0]] << "\t" << p[measuring_node[1]] << "\t" << p[measuring_node[2]] << "\t" << p[measuring_node[3]] << "\t" << maxh_h2 << "\t" << maxh_h4 << endl;
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
        ofile << i * dt << "\t" << p[measuring_node[0]] << "\t" << p[measuring_node[1]] << "\t" << p[measuring_node[2]] << "\t" << p[measuring_node[3]] << "\t" << maxh_h2 << "\t" << maxh_h4 << endl;
        ofile.close();
    }

    *file_write_control_1 = false;
}
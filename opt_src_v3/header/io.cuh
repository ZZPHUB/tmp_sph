#ifndef __IO_H__
#define __IO_H__

#include <atomic>

extern void ini_fluid(float* h_pos, float* h_vel, float* h_ac, float* rhop, float* p, int* particle_type, int* particle_zone, int np);
extern void output_fluid_file(int i,int np,float dt,float dx, float* h_pos, float* h_vel, float* p,int *table,int *h_particle_type, int* measuring_node);
extern void thread_loop(int *t_timestep,int t_np,float t_dt,float t_dx,float *t_pos,float *t_vel,float *t_p,int *t_table,int *t_type,int *t_node,std::atomic<int> *t_flag);

#endif
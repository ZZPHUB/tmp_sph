#ifndef __IO_H__
#define __IO_H__

extern void ini_fluid(float* h_pos, float* h_vel, float* h_ac, float* rhop, float* p, int* particle_type, int* particle_zone, int np);
extern void output_fluid_file(int cnt_frame, int i, float dt, int np, float* h_pos, float* h_vel, float* rhop, float* p, int* h_particle_type, int* h_particle_zone, int* h_vp_index, int* measuring_node, float dx, bool* file_write_control_1);

#endif
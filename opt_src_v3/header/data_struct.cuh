#ifndef __DATA_STRUCT_H__
#define __DATA_STRUCT_H__

typedef struct 
{
	///  simulation  --------
	float timeStep;
	float time;
	int numParticles;
	int boundary_p_num;

	//  gravity
	float gravity;

	//  Grid
	float gridSize, gridxmin, gridymin, gridzmin, gridxmax, gridymax, gridzmax;
	int gridxdim, gridydim, gridzdim, numCells;
	int hash_max, hash_min;
	float pre_set_domain_x_min, pre_set_domain_x_max, pre_set_domain_y_min, pre_set_domain_y_max, pre_set_domain_z_min, pre_set_domain_z_max;


	//  SPH  ---------------
	float particleR, h, kh, eta, delta, afa;		// smoothing radius
	float SpikyKern, LapKern, Poly6Kern;	// kernel consts
	float minDist;

	//  Fluid
	float particleMass, restDensity;
	float viscosity, stiffness;
	float cs;

	//kernel function
	float adh;

	//particle zone detection
	float namuta_threshold_1, namuta_threshold_2;

	//particle shifting
	float CFL, Ma, conc_0, conc_t, PST_R, PST_n, w_deltar;

	//solid imformation
	int ngp, nnode, n_edge, num_vp, max_nspnode, nele;
	float mat_para[3];
}SimParams;

#endif
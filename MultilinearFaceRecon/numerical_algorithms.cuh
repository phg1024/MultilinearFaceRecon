#pragma once

#include "Elements_GPU.h"
#include "utils_GPU.cuh"

namespace NumericalAlgorithms {
	extern float *x0, *x, *deltaX, *r, *J, *JtJ;
	void initialize(int nparams, int nconstraints);
	void finalize();

	__host__ int GaussNewton(int m, int n, int itmax, float *opts,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,		// feature points
		float *d_w_landmarks, float *d_w_mask, float w_fp_scale,	// weights for feature points
		d_ICPConstraint * d_icpc, int nicpc, float w_ICP,			// ICP terms
		float *d_tplt,												// template mesh
		cudaStream_t& mystream
		);
}
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include <cula.h>
#include <cublas.h>

#include "numerical_algorithms.cuh"
#include "Utils/cudautils.h"
#include "Utils/Timer.h"

#include "mkl.h"

namespace NumericalAlgorithms {
	float *x0, *x, *deltaX, *r, *J, *JtJ;
	float *h_JtJ, *h_Jtr;

	void initialize(int nparams, int nconstraints) {
		checkCudaErrors(cudaMalloc((void**)&x0, sizeof(float)*nparams));
		checkCudaErrors(cudaMalloc((void**)&x, sizeof(float)*nparams));
		checkCudaErrors(cudaMalloc((void**)&deltaX, sizeof(float)*nparams));

		checkCudaErrors(cudaMalloc((void**)&r, sizeof(float)*nconstraints));
		checkCudaErrors(cudaMalloc((void**)&J, sizeof(float)*nconstraints*nparams));
		checkCudaErrors(cudaMalloc((void**)&JtJ, sizeof(float)*nparams*nparams));

		h_JtJ = new float[nparams*nparams];
		h_Jtr = new float[nparams];
	}

	void finalize() {
		checkCudaErrors(cudaFree(x0));
		checkCudaErrors(cudaFree(x));
		checkCudaErrors(cudaFree(deltaX));
		checkCudaErrors(cudaFree(r));
		checkCudaErrors(cudaFree(J));
		checkCudaErrors(cudaFree(JtJ));

		delete[] h_JtJ;
		delete[] h_Jtr;
	}

	// use one dimensional configuration
	__global__ void cost_ICP(float *unknowns, float *costfunc, int offset, int step,
		d_ICPConstraint* d_icpc, int nicpc,
		float *d_tplt,
		float w_ICP
		) 
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= nicpc ) return;

		float s, rx, ry, rz, tx, ty, tz;
		rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
		tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
		s = unknowns[6];

		mat3 R = mat3::rotation(rx, ry, rz) * s;
		float3 T = make_float3(tx, ty, tz);

		d_ICPConstraint icpc = d_icpc[tid*step];
		const int3& v = icpc.v;
		const float3& bc = icpc.bcoords;

		int3 vidx = icpc.v*3;

		float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
		float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
		float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);
		float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;
		p = R * p + T;

		const float3& q = icpc.q;
		float3 pq = q - p;
		const float w_rigid = fabsf(icpc.weight)>1e-2?0.01:1.0;
		w_ICP *= w_rigid;

		costfunc[tid+offset] = dot(pq, pq) * w_ICP;
	}

	__global__ void jacobian_ICP(int m, float *unknowns, float *J, int offset, int step,
		d_ICPConstraint* d_icpc, int nicpc,
		float *d_tplt,
		float w_ICP) 
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= nicpc ) return;

		float s, rx, ry, rz, tx, ty, tz;
		rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
		tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
		s = unknowns[6];

		mat3 R = mat3::rotation(rx, ry, rz);
		float3 T = make_float3(tx, ty, tz);

		mat3 Jx, Jy, Jz;
		mat3::jacobian(rx, ry, rz, Jx, Jy, Jz);

		d_ICPConstraint icpc = d_icpc[tid*step];
		const int3& v = icpc.v;
		const float3& bc = icpc.bcoords;

		int3 vidx = icpc.v*3;

		float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
		float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
		float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);

		const float3& q = icpc.q;

		const float w_rigid = fabsf(icpc.weight)>1e-2?0.01:1.0;
		w_ICP *= w_rigid;

		float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;

		int jidx = (tid+offset)*m;

		// R * p
		//float3 rp = R * p;
		float3 rp = p;
		rotate_point(R, rp.x, rp.y, rp.z);

		// s * R * p + t - q
		float3 rk = s * rp + T - q;

		//float3 jp = Jx * p;
		float3 jp = p;
		rotate_point(Jx, jp.x, jp.y, jp.z);

		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;	

		//jp = Jy * p;
		jp = p;
		rotate_point(Jy, jp.x, jp.y, jp.z);
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

		//jp = Jz * p;
		rotate_point(Jz, jp.x, jp.y, jp.z);
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * rk.x * w_ICP;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * rk.y * w_ICP;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * rk.z * w_ICP;

		// fixed scale
		if( m < 7 ) return;
		// \frac{\partial r_i}{\partial s}
		J[jidx++] = 2.0 * dot(rp, rk) * w_ICP;
	}

	// use one dimensional configuration
	/* @note	d_w_mask is		1.0				if i<42 || i > 74
	w_outer * w_fp	if 63 < i <= 74
	w_chin * w_fp	if 42 <= i <= 63
	*/
	__global__ void cost_FeaturePoints(float *unknowns, float *costfunc, int offset,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,
		float *d_tplt,
		float *d_w_landmarks, float *d_w_mask,
		float w_fp_scale) 
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= nfpts ) return;

		float s, rx, ry, rz, tx, ty, tz;
		rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
		tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
		s = unknowns[6];

		mat3 R = mat3::rotation(rx, ry, rz) * s;
		float3 T = make_float3(tx, ty, tz);

		int voffset = tid * 3;
		float wpt = d_w_landmarks[tid] * w_fp_scale * d_w_mask[tid];

		int vidx = d_fptsIdx[tid] * 3;
		float3 p = make_float3(d_tplt[vidx], d_tplt[vidx+1], d_tplt[vidx+2]);

		p = R * p + T;

		if( tid < 42 || tid > 74 ) {
			float3 q = make_float3(d_q[voffset], d_q[voffset+1], d_q[voffset+2]);
			float3 pq = p-q;
			costfunc[tid+offset] = dot(p-q, p-q)*wpt;
		}
		else {
			float3 q = make_float3(d_q2d[voffset], d_q2d[voffset+1], d_q2d[voffset+2]);
			float3 uvd = world2color(p);
			float du = uvd.x - q.x, dv = uvd.y - q.y;
			costfunc[tid+offset] = (du*du+dv*dv)*wpt;
		}
	}

	__global__ void jacobian_FeaturePoints(int m, float *unknowns, float *J, int offset,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,
		float *d_tplt,
		float *d_w_landmarks, float *d_w_mask,
		float w_fp_scale)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= nfpts ) return;

		float s, rx, ry, rz, tx, ty, tz;
		rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
		tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
		s = unknowns[6];

		mat3 R = mat3::rotation(rx, ry, rz);
		float3 T = make_float3(tx, ty, tz);
		mat3 Jx, Jy, Jz;
		mat3::jacobian(rx, ry, rz, Jx, Jy, Jz);

		int voffset = tid * 3;
		float wpt = d_w_landmarks[tid] * w_fp_scale * d_w_mask[tid];

		int vidx = d_fptsIdx[tid] * 3;
		int jidx = (tid+offset)*m;
		float3 p = make_float3(d_tplt[vidx], d_tplt[vidx+1], d_tplt[vidx+2]);

		if( tid < 42 || tid > 74 ) {
			float3 q = make_float3(d_q[voffset], d_q[voffset+1], d_q[voffset+2]);

			// R * p
			float3 rp = R * p;

			// s * R * p + t - q
			float3 rk = s * rp + T - q;

			float3 jp = Jx * p;
			// \frac{\partial r_i}{\partial \theta_x}
			J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;	

			jp = Jy * p;
			// \frac{\partial r_i}{\partial \theta_y}
			J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;

			jp = Jz * p;
			// \frac{\partial r_i}{\partial \theta_z}
			J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;

			// \frac{\partial r_i}{\partial \t_x}
			J[jidx++] = 2.0 * rk.x * wpt;

			// \frac{\partial r_i}{\partial \t_y}
			J[jidx++] = 2.0 * rk.y * wpt;

			// \frac{\partial r_i}{\partial \t_z}
			J[jidx++] = 2.0 * rk.z * wpt;

			// fixed scale
			if( m < 7 ) return;
			// \frac{\partial r_i}{\partial s}
			J[jidx++] = 2.0 * dot(rp, rk) * wpt;
		}
		else {
			float3 q = make_float3(d_q2d[voffset], d_q2d[voffset+1], d_q2d[voffset+2]);

			float3 rp = R * p;
			float3 pk = s * rp + T;

			float inv_z = 1.0 / pk.z;
			float inv_z2 = inv_z * inv_z;

			const float f_x = 525.0, f_y = 525.0;
			float Jf[6] = {0};
			Jf[0] = -f_x * inv_z; Jf[2] = f_x * pk.x * inv_z2;
			Jf[4] = f_y * inv_z; Jf[5] = -f_y * pk.y * inv_z2;

			float3 uvd = world2color(pk);
			float pu = uvd.x, pv = uvd.y, pd = uvd.z;

			// residue
			float rkx = pu - q.x, rky = pv - q.y;

			// J_? * p_k
			float3 jp = Jx * p;
			// J_f * J_? * p_k
			float jfjpx, jfjpy;

			jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
			jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
			// \frac{\partial r_i}{\partial \theta_x}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jp = Jy * p;
			jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
			jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
			// \frac{\partial r_i}{\partial \theta_y}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jp = Jz * p;
			jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
			jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
			// \frac{\partial r_i}{\partial \theta_z}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_x}
			J[jidx++] = 2.0 * (Jf[0] * rkx) * wpt;

			// \frac{\partial r_i}{\partial \t_y}
			J[jidx++] = 2.0 * (Jf[4] * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_z}
			J[jidx++] = 2.0 * (Jf[2] * rkx + Jf[5] * rky) * wpt;

			// fixed scale
			if( m < 7 ) return;
			// \frac{\partial r_i}{\partial s}
			jfjpx = Jf[0] * rp.x + Jf[2] * rp.z;
			jfjpy = Jf[4] * rp.y + Jf[5] * rp.z;
			J[jidx++] = 2.0 * (jfjpx * rkx + jfjpy * rky) * wpt;
		}
	}

	__global__ void cost_History(float *unknowns, float *costfunc, int offset, float *d_meanRT, float w_history) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= 7 ) return;

		float diff = unknowns[tid] - d_meanRT[tid];
		costfunc[offset+tid] = diff * diff * w_history;
	}

	__global__ void jacobian_History(int m,float *unknowns, float *J, int offset, float *d_meanRT, float w_history) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= m ) return;

		float diff = unknowns[tid] - d_meanRT[tid];

		int jidx = (offset+tid)*m;
		for(int j=0;j<m;j++) {
			J[jidx+j] = 0;
		}
		J[jidx+tid] = 2.0 * diff * w_history;
	}

	/* Gaussian-Newton algorithm for estimating rigid transformation */
	__host__ int GaussNewton(int m, int n, int itmax, float *opts,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,		// feature points
		float *d_w_landmarks, float *d_w_mask, float w_fp_scale,	// weights for feature points
		d_ICPConstraint * d_icpc, int nicpc, float w_ICP,			// ICP terms
		float *d_meanRT, float w_history,							// history term
		float *d_tplt,												// template mesh
		cudaStream_t& mystream
		) 
	{	
		// setup parameters
		float delta, R_THRES, DIFF_THRES;
		if( opts == NULL ) {
			// use default values
			delta = 1.0;	// step size, default to use standard Newton-Ralphson
			R_THRES = 1e-6;	DIFF_THRES = 1e-6;
		}
		else {
			delta = opts[0]; R_THRES = opts[1]; DIFF_THRES = opts[2];
		}

		cudaMemcpy(deltaX, x, sizeof(float)*m, cudaMemcpyDeviceToDevice);
		checkCudaState();

		//cout << nicpc << endl;
		// compute initial residue with GPU
		dim3 block(1024, 1, 1);
		dim3 grid((int)(ceil(nicpc/(float)block.x)), 1, 1);
		cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, 1, d_icpc, nicpc, d_tplt, w_ICP);
		checkCudaState();
		cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
		checkCudaState();
		cost_History<<<1, 16>>>(x, r, nicpc+nfpts, d_meanRT, w_history);
		checkCudaState();
		//writeback(r, nicpc+nfpts, 1, "d_r.txt");

		//writeback(d_w_landmarks, nfpts, 1, "d_w_landmarks.txt");
		//writeback(d_w_mask, nfpts, 1, "d_w_mask.txt");
		//cout << "w_ICP = " << w_ICP << endl;
		//cout << "w_fp_scale = " << w_fp_scale << endl;

		float dstep = (delta - 0.1) / itmax;
		int iters = 0;
		PhGUtils::Timer t;
		// while not converged
		while( cublasSnrm2(m, deltaX, 1) > DIFF_THRES && cublasSnrm2(nicpc+nfpts+m, r, 1) > R_THRES && iters < itmax ) {
			//// compute jacobian with GPU
			//t.tic();
			jacobian_ICP<<<grid, block, 0, mystream>>>(m, x, J, 0, 1, d_icpc, nicpc, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("jacobian ICP");
			checkCudaState();
			//t.tic();
			jacobian_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(m, x, J, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("jacobian feature points");
			checkCudaState();

			jacobian_History<<<1, 16, 0, mystream>>>(m, x, J, nicpc+nfpts, d_meanRT, w_history);

			//writeback(J, nicpc+nfpts, m, "d_J.txt");

			//// store old values			
			//cudaMemcpy(x0, x, sizeof(float)*m, cudaMemcpyDeviceToDevice);
			checkCudaState();

			//// compute JtJ
			//t.tic();
			cublasSsyrk('U', 'N', m, nicpc+nfpts+m, 1.0, J, m, 0, JtJ, m);
			//cudaThreadSynchronize();
			//t.toc("JtJ");
			//writeback(JtJ, m, m, "d_JtJ.txt");

			//// compute Jtr
			//t.tic();
			cublasSgemv('N', m, nicpc+nfpts+m, 1.0, J, m, r, 1, 0, deltaX, 1);
			//cudaThreadSynchronize();
			//t.toc("Jtr");
			//writeback(deltaX, 7, 1, "d_Jtr.txt");

			//// compute deltaX
			
			//t.tic();
#if 0
			culaDeviceSpotrf('U', m, JtJ, m);
			culaDeviceSpotrs('U', m, 1, JtJ, m, deltaX, m);
			t.toc("JtJ\\Jtr");
#else
			cudaMemcpy(h_JtJ, JtJ, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Jtr, deltaX, sizeof(float)*m, cudaMemcpyDeviceToHost);

			LAPACKE_spotrf( LAPACK_COL_MAJOR, 'U', m, h_JtJ, m );
			LAPACKE_spotrs( LAPACK_COL_MAJOR, 'U', m, 1, h_JtJ, m, h_Jtr, m );

			cudaMemcpy(deltaX, h_Jtr, sizeof(float)*m, cudaMemcpyHostToDevice);
			//t.toc("h_JtJ\\h_Jtr");
#endif
			//cudaThreadSynchronize();
			//writeback(deltaX, 7, 1, "d_deltaX.txt");

			//// update x
			//t.tic();
			cublasSaxpy(m, -delta, deltaX, 1, x, 1);
			//t.toc("x <- x + dx");

			//// update residue
			//t.tic();
			cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, 1, d_icpc, nicpc, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("cost ICP");
			checkCudaState();
			//t.tic();
			cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("cost feature point");
			checkCudaState();
	
			cost_History<<<1, 16, 0, mystream>>>(x, r, nicpc+nfpts, d_meanRT, w_history);
			checkCudaState();

			//::system("pause");
			iters++;
			delta -= dstep;
		}

		//::system("pause");
		checkCudaState();
		return iters;
	}

	
	/* Gaussian-Newton algorithm for estimating rigid transformation */
	__host__ int GaussNewton_fp(int m, int n, int itmax, float *opts,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,		// feature points
		float *d_w_landmarks, float *d_w_mask, float w_fp_scale,	// weights for feature points
		d_ICPConstraint * d_icpc, int nicpc, float w_ICP,			// ICP terms
		float *d_meanRT, float w_history,							// history term
		float *d_tplt,												// template mesh
		cudaStream_t& mystream
		) 
	{	
		int frac = 1;
		// setup parameters
		float delta, R_THRES, DIFF_THRES;
		if( opts == NULL ) {
			// use default values
			delta = 1.0;	// step size, default to use standard Newton-Ralphson
			R_THRES = 1e-6;	DIFF_THRES = 1e-6;
		}
		else {
			delta = opts[0]; R_THRES = opts[1]; DIFF_THRES = opts[2];
		}

		cudaMemcpy(deltaX, x, sizeof(float)*m, cudaMemcpyDeviceToDevice);
		checkCudaState();

		int nicpcDfrac = nicpc/frac;
		int nr = nicpcDfrac+nfpts;
		float inv_nr = 1.0 / nr;

		//cout << nicpc << endl;
		// compute initial residue with GPU
		dim3 block(1024, 1, 1);
		dim3 grid((int)(ceil(nicpc/(float)block.x)), 1, 1);
		cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, frac, d_icpc, nicpcDfrac, d_tplt, w_ICP);
		checkCudaState();
		cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpcDfrac, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
		checkCudaState();
		cost_History<<<1, 16>>>(x, r, nicpc+nfpts, d_meanRT, w_history);
		checkCudaState();
		//writeback(r, nicpc+nfpts, 1, "d_r.txt");

		//writeback(d_w_landmarks, nfpts, 1, "d_w_landmarks.txt");
		//writeback(d_w_mask, nfpts, 1, "d_w_mask.txt");
		//cout << "w_ICP = " << w_ICP << endl;
		//cout << "w_fp_scale = " << w_fp_scale << endl;
		
		float dstep = (delta - 0.05) / itmax;
		int iters = 0;
		PhGUtils::Timer t;
		// while not converged
		while( cublasSnrm2(m, deltaX, 1) > DIFF_THRES && cublasSnrm2(nr+m, r, 1)*inv_nr > R_THRES && iters < itmax ) {
			//// compute jacobian with GPU
			//t.tic();
			jacobian_ICP<<<grid, block, 0, mystream>>>(m, x, J, 0, frac, d_icpc, nicpcDfrac, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("jacobian ICP");
			checkCudaState();
			//t.tic();
			jacobian_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(m, x, J, nicpcDfrac, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("jacobian feature points");
			checkCudaState();

			jacobian_History<<<1, 16, 0, mystream>>>(m, x, J, nr, d_meanRT, w_history);

			//writeback(J, nicpc+nfpts, m, "d_J.txt");

			//// store old values			
			//cudaMemcpy(x0, x, sizeof(float)*m, cudaMemcpyDeviceToDevice);
			checkCudaState();

			//// compute JtJ
			//t.tic();
			cublasSsyrk('U', 'N', m, nr+m, 1.0, J, m, 0, JtJ, m);
			//cudaThreadSynchronize();
			//t.toc("JtJ");
			//writeback(JtJ, m, m, "d_JtJ.txt");

			//// compute Jtr
			//t.tic();
			cublasSgemv('N', m, nr+m, 1.0, J, m, r, 1, 0, deltaX, 1);
			//cudaThreadSynchronize();
			//t.toc("Jtr");
			//writeback(deltaX, 7, 1, "d_Jtr.txt");

			//// compute deltaX
			
			//t.tic();
#if 1
			culaDeviceSpotrf('U', m, JtJ, m);
			culaDeviceSpotrs('U', m, 1, JtJ, m, deltaX, m);
			//t.toc("JtJ\\Jtr");
#else
			cudaMemcpy(h_JtJ, JtJ, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_Jtr, deltaX, sizeof(float)*m, cudaMemcpyDeviceToHost);

			LAPACKE_spotrf( LAPACK_COL_MAJOR, 'U', m, h_JtJ, m );
			LAPACKE_spotrs( LAPACK_COL_MAJOR, 'U', m, 1, h_JtJ, m, h_Jtr, m );

			cudaMemcpy(deltaX, h_Jtr, sizeof(float)*m, cudaMemcpyHostToDevice);
			//t.toc("h_JtJ\\h_Jtr");
#endif
			//cudaThreadSynchronize();
			//writeback(deltaX, 7, 1, "d_deltaX.txt");

			//// update x
			//t.tic();
			cublasSaxpy(m, -delta, deltaX, 1, x, 1);
			//t.toc("x <- x + dx");

			//// update residue
			//t.tic();
			cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, frac, d_icpc, nicpcDfrac, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("cost ICP");
			checkCudaState();
			//t.tic();
			cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpcDfrac, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("cost feature point");
			checkCudaState();
	
			cost_History<<<1, 16, 0, mystream>>>(x, r, nr, d_meanRT, w_history);
			checkCudaState();

			//::system("pause");
			iters++;
			delta -= dstep;
		}

		//::system("pause");
		checkCudaState();
		return iters;
	}
}

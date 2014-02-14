#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include <cula.h>
#include <cublas.h>

#include "numerical_algorithms.cuh"
#include "Utils/cudautils.h"
#include "Utils/Timer.h"

namespace NumericalAlgorithms {
	float *x0, *x, *deltaX, *r, *J, *JtJ;

	void initialize(int nparams, int nconstraints) {
		checkCudaErrors(cudaMalloc((void**)&x0, sizeof(float)*nparams));
		checkCudaErrors(cudaMalloc((void**)&x, sizeof(float)*nparams));
		checkCudaErrors(cudaMalloc((void**)&deltaX, sizeof(float)*nparams));

		checkCudaErrors(cudaMalloc((void**)&r, sizeof(float)*nconstraints));
		checkCudaErrors(cudaMalloc((void**)&J, sizeof(float)*nconstraints*nparams));
		checkCudaErrors(cudaMalloc((void**)&JtJ, sizeof(float)*nparams*nparams));
	}

	void finalize() {
		checkCudaErrors(cudaFree(x0));
		checkCudaErrors(cudaFree(x));
		checkCudaErrors(cudaFree(deltaX));
		checkCudaErrors(cudaFree(r));
		checkCudaErrors(cudaFree(J));
		checkCudaErrors(cudaFree(JtJ));
	}

	// use one dimensional configuration
	__global__ void cost_ICP(float *unknowns, float *costfunc, int offset,
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

		d_ICPConstraint icpc = d_icpc[tid];
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
		costfunc[tid+offset] = dot(pq, pq) * w_ICP;
	}

	__global__ void jacobian_ICP(float *unknowns, float *J, int offset,
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

		d_ICPConstraint icpc = d_icpc[tid];
		const int3& v = icpc.v;
		const float3& bc = icpc.bcoords;

		int3 vidx = icpc.v*3;

		float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
		float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
		float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);

		const float3& q = icpc.q;

		float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;

		int jidx = (tid+offset)*7;

		// R * p
		float3 rp = R * p;

		// s * R * p + t - q
		float3 rk = s * rp + T - q;

		float3 jp = Jx * p;
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;	

		jp = Jy * p;
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

		jp = Jz * p;
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * rk.x * w_ICP;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * rk.y * w_ICP;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * rk.z * w_ICP;

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

	__global__ void jacobian_FeaturePoints(float *unknowns, float *J, int offset,
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
		int jidx = (tid+offset)*7;
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

			// \frac{\partial r_i}{\partial s}
			jfjpx = Jf[0] * rp.x + Jf[2] * rp.z;
			jfjpy = Jf[4] * rp.y + Jf[5] * rp.z;
			J[jidx++] = 2.0 * (jfjpx * rkx + jfjpy * rky) * wpt;
		}
	}

	__global__ void cost_History() {
	}

	__global__ void jacobian_History() {
	}

	/* Gaussian-Newton algorithm for estimating rigid transformation */
	__host__ int GaussNewton(int m, int n, int itmax, float *opts,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,		// feature points
		float *d_w_landmarks, float *d_w_mask, float w_fp_scale,	// weights for feature points
		d_ICPConstraint * d_icpc, int nicpc, float w_ICP,			// ICP terms
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
		dim3 block(256, 1, 1);
		dim3 grid((int)(ceil(nicpc/(float)block.x)), 1, 1);
		cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, d_icpc, nicpc, d_tplt, w_ICP);
		checkCudaState();
		cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
		checkCudaState();
		//writeback(r, nicpc+nfpts, 1, "d_r.txt");

		//writeback(d_w_landmarks, nfpts, 1, "d_w_landmarks.txt");
		//writeback(d_w_mask, nfpts, 1, "d_w_mask.txt");
		//cout << "w_ICP = " << w_ICP << endl;
		//cout << "w_fp_scale = " << w_fp_scale << endl;

		int iters = 0;
		//PhGUtils::Timer t;
		// while not converged
		while( cublasSnrm2(m, deltaX, 1) > DIFF_THRES && cublasSnrm2(n, r, 1) > R_THRES && iters < itmax ) {
			//// compute jacobian with GPU
			//t.tic();
			jacobian_ICP<<<grid, block, 0, mystream>>>(x, J, 0, d_icpc, nicpc, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("jacobian ICP");
			checkCudaState();
			//t.tic();
			jacobian_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, J, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("jacobian feature points");
			checkCudaState();

			//writeback(J, nicpc+nfpts, 7, "d_J.txt");

			//// store old values			
			//cudaMemcpy(x0, x, sizeof(float)*m, cudaMemcpyDeviceToDevice);
			checkCudaState();

			//// compute JtJ
			//t.tic();
			cublasSsyrk('U', 'N', m, n, 1.0, J, m, 0, JtJ, m);
			//cudaThreadSynchronize();
			//t.toc("JtJ");
			//writeback(JtJ, 7, 7, "d_JtJ.txt");

			//// compute Jtr
			//t.tic();
			cublasSgemv('N', m, n, 1.0, J, m, r, 1, 0, deltaX, 1);
			//cudaThreadSynchronize();
			//t.toc("Jtr");
			//writeback(deltaX, 7, 1, "d_Jtr.txt");

			//// compute deltaX
			
			//t.tic();
			culaDeviceSpotrf('U', m, JtJ, m);
			culaDeviceSpotrs('U', m, 1, JtJ, m, deltaX, m);
			//cudaThreadSynchronize();
			//t.toc("JtJ\\Jtr");
			//writeback(deltaX, 7, 1, "d_deltaX.txt");

			//// update x
			//t.tic();
			cublasSaxpy(m, -delta, deltaX, 1, x, 1);
			//t.toc("x <- x + dx");

			//// update residue
			//t.tic();
			cost_ICP<<<grid, block, 0, mystream>>>(x, r, 0, d_icpc, nicpc, d_tplt, w_ICP);
			//cudaThreadSynchronize();
			//t.toc("cost ICP");
			checkCudaState();
			//t.tic();
			cost_FeaturePoints<<<dim3(1, 1, 1), dim3(128, 1, 1), 0, mystream>>>(x, r, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp_scale);
			//cudaThreadSynchronize();
			//t.toc("cost feature point");
			checkCudaState();

			//::system("pause");
			iters++;
		}

		//::system("pause");
		checkCudaState();
		return iters;
	}
}

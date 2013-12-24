#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"

#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/cudautils.h"

#include <cula.h>
#include <cublas.h>

class MultilinearReconstructorGPU 
{
public:
	MultilinearReconstructorGPU();
	~MultilinearReconstructorGPU();

protected:
	void init();
	void preprocess();

private:
	// host resources
	// convergence criteria
	float cc;
	float errorThreshold, errorDiffThreshold;
	static const int MAXITERS = 8;
	bool usePrior;

	// weights for prior
	float w_prior_id, w_prior_exp;
	float w_boundary;

	// template face
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;

	float RTparams[7]; /* sx, ry, rz, tx, ty, tz, scale */
	
	bool useHistory;
	static const int historyLength = 5;
	float historyWeights[historyLength];
	deque<vector<float>> RTHistory;	// stores the RT parameters for last 5 frames
	
	// used to avoid local minima
	float meanX, meanY, meanZ;
	float scale;

	// fitting control
	static const int INITFRAMES = 5;
	int frameCounter;
	bool fitPose, fitIdentity, fitExpression;

private:
	// the input core tensor
	int core_dim[3];

	// the unfolded tensor
	float* d_tu0, *d_tu1;

	// the tensor after mode product
	float* d_tm0, *d_tm1;

	vector<int> landmarkIdx;

	// the truncated core tensor, unfolded
	float* d_tu0c, *d_tu1c;
		
	// the tensor after mode product, with truncation
	// they are changed ONLY if wid or wexp is changed
	// tm0c: corec mode product with wid
	// tm1c: corec mode product with wexp
	float* d_tm0c, *d_tm1c;

	// the tensor after mode product, with truncation, and after rigid transformation
	// they are changed ONLY if the rigid transformation changes
	// tm0c: corec mode product with wid, after rigid transformation
	// tm1c: corec mode product with wexp, after rigid transformation
	float* d_tm0cRT, *d_tm1cRT;

	// the tensor after 2 mode products, with truncation; 
	// tmc is the tensor before applying global transformation
	float* d_tmc;
	
	float* d_q;			// target point coordinates

	int* d_fptsIdx;
	float* d_targets;

	// computation weights
	float* d_w_landmarks;

	float *d_RTparams;

	// weights prior
	float *d_mu_wid0, *d_mu_wexp0;
	float *d_mu_wid, *d_mu_wexp;
	float *d_mu_wid_weighted, *d_mu_wexp_weighted;
	int ndims_wid, ndims_wexp;
	float *d_sigma_wid, *d_sigma_wexp;
	float *d_sigma_wid_weighted, *d_sigma_wexp_weighted;

	// weights
	float *d_Wid, *d_Wexp;

	// computation matrices
	float *d_Aid, *d_Aexp;
	float *d_brhs;
	float *d_AidtAid, *d_AexptAexp;
	float *d_Aidtb, *d_Aexptb;
};
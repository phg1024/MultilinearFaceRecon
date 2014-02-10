#pragma once

#include "phgutils.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda_gl.h>


#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
#include "Geometry/Mesh.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/MeshWriter.h"

#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/cudautils.h"

#include <cula.h>
#include <cublas.h>

#include <QGLWidget>
#include <QGLFramebufferObject>

struct d_ICPConstraint {
	int3 v;					// incident vertices
	float3 bcoords;			// barycentric coordiantes
	float weight;			
	float3 q;				// target point
};

class MultilinearReconstructorGPU 
{
public:
	enum FittingOption {
		FIT_POSE,
		FIT_IDENTITY,
		FIT_EXPRESSION,
		FIT_POSE_AND_IDENTITY,
		FIT_POSE_AND_EXPRESSION,
		FIT_ALL
	};

	MultilinearReconstructorGPU();
	~MultilinearReconstructorGPU();

	void setPose(const float* params);
	void setIdentityWeights(const Tensor1<float>& t);
	void setExpressionWeights(const Tensor1<float>& t);

	// bind 3D landmarks
	void bindTarget(const vector<PhGUtils::Point3f>& tgt);
	void bindRGBDTarget(const vector<unsigned char>& colordata,
						const vector<unsigned char>& depthdata);

	void fit(FittingOption op);
	void fit_featurepoints();
	void fitPose();
	void fitPoseAndIdentity();
	void fitPoseAndExpression();
	void fitAll();

	void setBaseMesh(const PhGUtils::QuadMesh& m) { baseMesh = m; }
	const Tensor1<float>& currentMesh() { return tmesh; }

protected:
	void init();
	void preprocess();
	void initRenderer();

protected:
	int collectICPConstraints(int, int);
	bool fitRigidTransformation();

	float computeError();

	void transformMesh();	// transform the mesh on the gpu side
	void updateMesh();		// copy back the generated mesh and update the current mesh on CPU side

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

	// fitted face
	Tensor1<float> tmesh;

	float h_RTparams[7]; /* sx, ry, rz, tx, ty, tz, scale */
	
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

private:
	// computation stream
	cudaStream_t mystream;

	// the input core tensor
	int core_dim[3];

	// the unfolded tensor
	float *d_tu0, *d_tu1;

	// the tensor after mode product
	float *d_tm0, *d_tm1;

	// template tensor: d_tplt = core x_1 Wid x_2 Wexp
	float *d_tplt;
	float *d_mesh;

	vector<int> landmarkIdx;

	// the tensor after mode product, with truncation, and after rigid transformation
	// they are changed ONLY if the rigid transformation changes
	// tm0c: corec mode product with wid, after rigid transformation
	// tm1c: corec mode product with wexp, after rigid transformation
	float *d_tm0RT, *d_tm1RT;

	int* d_fptsIdx;
	float *h_q2d, *d_q2d;		// 2d landmarks
	float *h_q, *d_q;			// 3d landmarks

	// weights of the feature points
	float *h_w_landmarks, *d_w_landmarks;

	// raw color / depth data, need to perform conversion before using them
	unsigned char *d_colordata, *d_depthdata;	// image data on device

	d_ICPConstraint* d_icpc;
	float *d_targetLocations;

	float *d_RTparams;

	// weights prior
	float *d_mu_wid0, *d_mu_wexp0;	// neutral face weights
	float *d_mu_wid, *d_mu_wexp;	// blendshape weights
	float *d_mu_wid_weighted, *d_mu_wexp_weighted;	// blendshape weights weighted by given prior weights
	int ndims_wid, ndims_wexp, ndims_pts, ndims_fpts;
	int nfpts, npixels;

	float *d_sigma_wid, *d_sigma_wexp;
	float *d_sigma_wid_weighted, *d_sigma_wexp_weighted;

	// weights
	float *d_Wid, *d_Wexp;

	// computation matrices
	float *d_A, *d_b;

private:
	void renderMesh();

	PhGUtils::QuadMesh baseMesh;
	PhGUtils::Matrix4x4f mProj, mMv;

	shared_ptr<QGLWidget> dummyWgt;
	shared_ptr<QGLFramebufferObject> fbo;

	vector<float> depthMap;						// synthesized depth map
	vector<unsigned char> indexMap;				// synthesized face index map

	// GPU copy of the synthesized depth and index
	unsigned char* d_indexMap;
	float* d_depthMap;
};
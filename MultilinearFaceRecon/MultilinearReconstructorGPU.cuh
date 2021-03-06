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
#include "Utils/Timer.h"

#include "MRSetting.h"

#include <cula.h>
#include <cublas.h>

#include <QGLWidget>
#include <QGLShaderProgram>
#define USE_QT_FBO 0
#if USE_QT_FBO
#include <QGLFramebufferObject>
#else
#include "OpenGL/fbo.h"
#endif

#include "Elements_GPU.h"

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

	const float* getPose() const { return h_RTparams; }
	const Tensor1<float>& expressionWeights() const { return Wexp; }
	const Tensor1<float>& identityWeights() const { return Wid; }

	float reconstructionError() const {
		return E;
	}

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

	void setBaseMesh(const PhGUtils::QuadMesh& m);
	const Tensor1<float>& currentMesh() const { return tmesh; }

protected:
	void init();
	void initializeWeights();
	void preprocess();
	void initRenderer();

protected:
	int collectICPConstraints(int, int);
	bool fitRigidTransformation(bool fitScale, int&);
	bool fitIdentityWeights();
	bool fitExpressionWeights();

	float computeError();
	vector<float> computeWeightedMeanPose();

	void transformTM0();
	void transformTM1();

	void transformMesh();	// transform the mesh on the gpu side
	void updateMesh();		// copy back the generated mesh and update the current mesh on CPU side

private:
	// host resources
	// convergence criteria
	float cc;
	float errorThreshold, errorDiffThreshold;
	static const int MAXITERS = 16;
	bool usePrior;

	// weights for prior
	float w_prior_id, w_prior_exp;
	float w_boundary;

	// fitted face
	Tensor1<float> tmesh;

	float h_RTparams[7]; /* rx, ry, rz, tx, ty, tz, scale */
	Tensor1<float> Wid, Wexp;

	bool useHistory;
	static const int historyLength = 10;
	float historyWeights[historyLength];
	deque<vector<float>> RTHistory;	// stores the RT parameters for last 5 frames
	
	// used to avoid local minima
	float meanX, meanY, meanZ;
	float scale;

	// fitting control
	static const int INITFRAMES = 5;
	int frameCounter;

	float E;
private:
	// computation stream
	cudaStream_t mystream;

	// the input core tensor
	int core_dim[3];

	// the unfolded tensor
	float *h_tu0, *h_tu1;	// host side page-locked memory
	float *d_tu0, *d_tu1;	// device mapped address

	// the tensor after mode product
	float *d_tm0, *d_tm1;

	// template tensor: d_tplt = core x_1 Wid x_2 Wexp
	float *d_tplt;
	float *d_mesh;
	float *d_rawdistmap, *d_distmap;

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
	float *h_w_landmarks_exp, *d_w_landmarks_exp;	// for expression fitting
	float *h_w_landmarks_rigid, *d_w_landmarks_rigid;

	// raw color / depth data, need to perform conversion before using them
	unsigned char *d_colordata, *d_depthdata;	// image data on device

	d_ICPConstraint *d_icpc, *d_icpc_rigid;
	int *d_icpc_rigid_idx;			// indices of icpc for pose estimation
	int *d_nicpc, *d_nicpc_rigid;
	int nicpc, nicpc_rigid;
	static const int MAX_ICPC_COUNT = 32768;
	float *d_targetLocations;

	float *d_RTparams;
	float *d_meanRT;

	// weights prior
	float *d_mu_wid0, *d_mu_wexp0;	// neutral face weights
	float *d_mu_wid, *d_mu_wexp;	// blendshape weights
	float *d_mu_wid_weighted, *d_mu_wexp_weighted;	// blendshape weights weighted by given prior weights
	int ndims_wid, ndims_wexp, ndims_pts, ndims_fpts;
	int npts_mesh, nfpts, npixels;

	float *d_sigma_wid, *d_sigma_wexp;
	//float *d_sigma_wid_weighted, *d_sigma_wexp_weighted;

	// weights
	float *d_Wid, *d_Wexp;

	// computation matrices
	double *d_A, *d_b;
	double *d_AtA, *d_Atb;

	// weights again
	float *d_w_mask, *h_w_mask;
	float w_ICP;				// weight for the ICP term (all constraints)
	float w_chin, w_outer;

	float w_fp, w_history;

	// error term
	float *d_error, *d_w_error;
	float *h_error, *h_w_error;

private:
	void renderMesh();

	PhGUtils::QuadMesh baseMesh;
	int4 *d_meshtopo;
	vector<unsigned int> h_meshtopo;
	//float3 *d_meshverts;
	vector<float3> h_meshverts;		// converted triangle mesh vertices
	int validfaces;
	vector<int> frontFaces;
	int *d_frontFaces;
	vector<bool> isBackFace;
	vector<float3> h_faceidx;		// converted triangle mesh face indices
	GLuint meshvbo, faceidxvbo, vao;
	cudaGraphicsResource_t vBuf;

	PhGUtils::Matrix4x4f mProj, mMv;

	shared_ptr<QGLWidget> dummyWgt;
	shared_ptr<QGLShaderProgram> shader;
#if USE_QT_FBO
	shared_ptr<QGLFramebufferObject> fbo;
#else
	shared_ptr<PhGUtils::FBO> fbo;
#endif
	cudaGraphicsResource_t cTex, dTex;		// color texture and depth texture
	cudaArray *cArray, *dArray;				// arrays to hold the mapped resources

	vector<float> depthMap;						// synthesized depth map
	vector<unsigned char> indexMap;				// synthesized face index map

	// GPU copy of the synthesized depth and index
	unsigned char* d_indexMap;
	float* d_depthMap;

	// stats
public:
	void printStats();

private:
	PhGUtils::Timer tTrans, tUpdate, tRender, tCollect, 
		tRigid, tTrans0, tExpr, tUpdate0, tError;
	vector<int> constraintCount;
	double totalCons;
	vector<int> constraintCount_rigid;
	double totalCons_rigid;
	vector<int> rigidIterations;
	double totalRigidIters;


private:
	MRSetting settings;
};
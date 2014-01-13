#pragma once

#include <QApplication>

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
#include "Geometry/Mesh.h"
#include "Geometry/MeshLoader.h"

// levmar header
#include "levmar.h"

#include <cula.h>

#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include <armadillo>

#include <QGLWidget>
#include <QGLFramebufferObject>

class MultilinearReconstructor : public QObject
{
	Q_OBJECT
public:
	enum TargetType {
		TargetType_2D,
		TargetType_3D
	};

	enum FittingOption {
		FIT_POSE,
		FIT_IDENTITY,
		FIT_EXPRESSION,
		FIT_POSE_AND_IDENTITY,
		FIT_POSE_AND_EXPRESSION,
		FIT_ALL
	};

	MultilinearReconstructor(void);
	~MultilinearReconstructor(void);

	// set the base mesh for synthesis
	void setBaseMesh(const PhGUtils::QuadMesh& m) {
		baseMesh = m;
	}

	// bind a fbo for synthesis
	void bindFBO(const shared_ptr<QGLFramebufferObject>& f) {
		fbo = f;
	}

	// reset the reconstructor
	void reset();

	float reconstructionError() const {
		return E;
	}

	const Tensor1<float>& templateMesh() const {
		return tplt;
	}

	const Tensor1<float>& currentMesh() const {
		return tmesh;
	}

	const float* getPose() const {
		return RTparams;
	}

	// for reconstruction with 2D feature points, first obtain
	// corresponding 3D locations, then use the reconstruction
	// for 3D points

	// for reconstruction with 3D locations of feature points
	void bindTarget(const vector<pair<PhGUtils::Point3f, int>>& pts,
		TargetType ttp = TargetType_3D);

	void bindRGBDTarget(
		const vector<unsigned char>& colordata,
		const vector<unsigned char>& depthdata
		);

	void init();
	void fit(FittingOption ops = FIT_ALL);
	void fit_withPrior();
	void fit2d(FittingOption ops = FIT_ALL);
	void fit2d_withPrior();

	void fitICP(FittingOption ops = FIT_ALL);
	void fitICP_withPrior();

	// fit a mesh given a path to file and a hint for initial fit
	// returns rotation, translation, scale, identity weights and expression weights
	tuple<vector<float>, vector<float>, vector<float>> fitMesh(const string& filename, const vector<pair<int, int>>& hint);

	// force update computation tensors
	void updateTM0() {
		tm0 = core.modeProduct(Wid, 0);
		tplt = tm0.modeProduct(Wexp, 0);
	}
	void updateTM1() {
		tm1 = core.modeProduct(Wexp, 1);
		tplt = tm1.modeProduct(Wid, 0);
	}

	void togglePrior();

	const Tensor1<float>& expressionWeights() const { return Wexp; }
	const Tensor1<float>& identityWeights() const { return Wid; }

	float expPriorWeights() const { return w_prior_exp; }
	void expPriorWeights(float val) { w_prior_exp = val; }
	float idPriorWeight() const { return w_prior_id; }
	void idPriorWeight(float val) { w_prior_id = val; }

signals:
	void oneiter();

private:
	void loadPrior();
	void loadCoreTensor();
	void createTemplateItem();
	void unfoldTensor();
	void initializeWeights();

	void updateComputationTensor();
	void updateMatrices();
	void updateCoreC();
	void updateTMC();
	void updateTMCwithTM0C();
	void transformTM0C();
	void transformTM1C();

	void updateTM();
	void updateTMwithTM0();
	void transformTM0();
	void transformTM1();

private:
	// reconstruction with 3D feature points
	friend void evalCost(float *p, float *hx, int m, int n, void* adata);
	friend void evalJacobian(float *p, float *J, int m, int n, void* adata);
	friend void evalCost2(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3(float *p, float *hx, int m, int n, void* adata);

	float computeError();

	bool fitRigidTransformationAndScale();
	bool fitIdentityWeights_withPrior();
	bool fitExpressionWeights_withPrior();

	// reconstruction with 2D feature points
	friend void evalCost_2D(float *p, float *hx, int m, int n, void* adata);
	friend void evalJacobian_2D(float *p, float *J, int m, int n, void* adata);
	friend void evalCost2_2D(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3_2D(float *p, float *hx, int m, int n, void* adata);

	float computeError_2D();

	bool fitRigidTransformationAndScale_2D();
	bool fitIdentityWeights_withPrior_2D();
	bool fitExpressionWeights_withPrior_2D();

	// reconstruction with ICP
	friend void evalCost_ICP(float *p, float *hx, int m, int n, void* adata);
	friend void evalJacobian_ICP(float *p, float *hx, int m, int n, void* adata);
	
	bool fitRigidTransformationAndScale_ICP();
	bool fitIdentityWeights_withPrior_ICP();
	bool fitExpressionWeights_withPrior_ICP();

	// reconstruction utilities
	void createFaceMask();
	void transformMesh();
	vector<float> computeWeightedMeanPose();

private:
	PhGUtils::QuadMesh baseMesh;
	PhGUtils::Matrix4x4f mProj, mMv;

	shared_ptr<QGLWidget> dummyWgt;
	shared_ptr<QGLFramebufferObject> fbo;
	
	vector<float> depthMap;						// synthesized depth map
	vector<unsigned char> indexMap;				// synthesized face index map

	vector<unsigned char> targetColor, targetDepth;	// target color and depth image
	vector<unsigned char> faceMask;					// mask over the target RGBD image
	vector<PhGUtils::Point3f> targetLocations;		// target locations obtain by back projecting color and depth image

	PhGUtils::DenseMatrix<float> Aid_ICP, Aexp_ICP;
	PhGUtils::DenseVector<float> brhs_ICP;

	float w_ICP;

	// for ICP
	struct ICPConstraint {
		ICPConstraint() {
			v[0] = -1, v[1] = -1, v[2] = -1;
		}
		ICPConstraint(const ICPConstraint& other) {
			for(int i=0;i<3;i++) {
				v[i] = other.v[i];
				bcoords[i] = other.bcoords[i];
			}
			weight = other.weight;
			q = other.q;
		}
		ICPConstraint& operator=(const ICPConstraint& other) {
			if( this != &other ) {
				for(int i=0;i<3;i++) {
					v[i] = other.v[i];
					bcoords[i] = other.bcoords[i];
				}
				weight = other.weight;
				q = other.q;
			}
			return (*this);
		}

		int v[3];				// incident vertices
		float bcoords[3];		// barycentric coordinates
		float weight;			// constraint weight
		PhGUtils::Point3f q;	// target point
	};
	vector<ICPConstraint> icpc;

private:
	void collectICPConstraints();
	void collectICPConstraints_topo();

	void updateMesh();
	void renderMesh();
private:
	// convergence criteria
	float cc;
	float errorThreshold, errorDiffThreshold;
	static const int MAXITERS = 32;	// this should be enough
	bool usePrior;

	// weights for prior
	float w_prior_id, w_prior_exp;
	// outer contour: 64~74
	// chin: 42~63
	float w_boundary, w_chin, w_outer;
	float w_fp;		// feature point weight when using ICP

	float w_prior_id_2D, w_prior_exp_2D;
	float w_boundary_2D;

	// the input core tensor
	Tensor3<float> core;
	// the truncated input core tensor
	Tensor3<float> corec;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;

	Tensor2<float> tm0RT, tm1RT;
		
	// the tensor after mode product, with truncation
	// they are changed ONLY if wid or wexp is changed
	// tm0c: corec mode product with wid
	// tm1c: corec mode product with wexp
	Tensor2<float> tm0c, tm1c;

	// the tensor after mode product, with truncation, and with rotation but not translation
	// they are changed ONLY if the rigid transformation changes
	// tm0cRT: corec mode product with wid, with rotation
	// tm1cRT: corec mode product with wexp, with rotation
	Tensor2<float> tm0cRT, tm1cRT;

	// the tensor after 2 mode products, with truncation; 
	// tmc is the tensor before applying global transformation
	Tensor1<float> tmc;
	
	Tensor1<float> q;			// target point coordinates

	// template face
	// the tensor after 2 mode products and before applying global transformation
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;

	// workspace for rigid fitting
	struct PoseWorkspace{
		vector<float> meas;
	} pws;

	float RTparams[7]; /* sx, ry, rz, tx, ty, tz, scale */
	float meanRT[7];
	
	bool useHistory;
	float w_history;
	static const int historyLength = 10;
	float historyWeights[historyLength];
	deque<vector<float>> RTHistory;	// stores the RT parameters for last 5 frames
	
	// used to avoid local minima
	float meanX, meanY, meanZ;
	float scale;
	arma::fmat R;
	arma::fvec T;
	PhGUtils::Matrix3x3f Rmat;
	PhGUtils::Point3f Tvec;

	// computation related
	PhGUtils::DenseMatrix<float> Aid, Aexp;
	PhGUtils::DenseMatrix<float> AidtAid, AexptAexp;
	PhGUtils::DenseVector<float> brhs, Aidtb, Aexptb;

	// weights
	Tensor1<float> Wid, Wexp;

	// weights prior
	// average identity and neutral expression
	arma::fvec mu_wid0, mu_wexp0;
	// mean wid and mean wexp
	arma::fvec mu_wid_orig, mu_wexp_orig;
	// mean wid and mean wexp, multiplied by the inverse of sigma_?
	arma::fvec mu_wid, mu_wexp;
	// weighted version of mu_wid and mu_wexp
	arma::fvec mu_wid_weighted, mu_wexp_weighted;
	arma::fmat sigma_wid, sigma_wexp;
	arma::fmat sigma_wid_weighted, sigma_wexp_weighted;

	// target vertices
	vector<pair<PhGUtils::Point3f, int>> targets;		// xyz, vertex index
	vector<pair<PhGUtils::Point3f, int>> targets_2d;	// uvd, vertex index
	vector<float> w_landmarks;

	// fitting control
	static const int INITFRAMES = 5;
	int frameCounter;
	bool fitScale, fitPose, fitIdentity, fitExpression;

	// fitting error
	float E;
};


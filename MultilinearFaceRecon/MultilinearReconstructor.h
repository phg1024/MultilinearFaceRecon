#pragma once

#include <QApplication>

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"

// levmar header
#include "levmar.h"

#define USE_CMINPACK 0
#if USE_CMINPACK
// cminpack header
#define __cminpack_float__
#define CMINPACK_NO_DLL
#include "../cminpack-1.3.2/cminpack.h"
#endif

#include <cula.h>

#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include <armadillo>

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
	void init();
	void fit(FittingOption ops = FIT_ALL);
	void fit_withPrior();
	void fit2d(FittingOption ops = FIT_ALL);
	void fit2d_withPrior();

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

private:
#if USE_CMINPACK
	friend int evalCost_minpack(void *adata, int m, int n, const __cminpack_real__ *p, __cminpack_real__ *hx,
		int iflag);
#endif
	friend void evalCost(float *p, float *hx, int m, int n, void* adata);
	friend void evalJacobian(float *p, float *J, int m, int n, void* adata);
	friend void evalCost2(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3(float *p, float *hx, int m, int n, void* adata);

	void transformMesh();
	float computeError();

	bool fitRigidTransformationAndScale();
	bool fitIdentityWeights_withPrior();
	bool fitExpressionWeights_withPrior();

	friend void evalCost_2D(float *p, float *hx, int m, int n, void* adata);
	friend void evalJacobian_2D(float *p, float *J, int m, int n, void* adata);
	friend void evalCost2_2D(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3_2D(float *p, float *hx, int m, int n, void* adata);

	bool fitRigidTransformationAndScale_2D();
	bool fitIdentityWeights_withPrior_2D();
	bool fitExpressionWeights_withPrior_2D();

	float computeError_2D();

	vector<float> computeWeightedMeanPose();

private:
	// convergence criteria
	float cc;
	float errorThreshold, errorDiffThreshold;
	static const int MAXITERS = 8;	// this should be enough
	bool usePrior;

	// weights for prior
	float w_prior_id, w_prior_exp;
	// outer contour: 64~74
	// chin: 42~63
	float w_boundary, w_chin;

	float w_prior_id_2D, w_prior_exp_2D;

	// the input core tensor
	Tensor3<float> core;
	// the truncated input core tensor
	Tensor3<float> corec;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;
		
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
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;

	// workspace for rigid fitting
	struct PoseWorkspace{
		vector<float> meas;
		// for cminpack
		//int workspace[npts];
		//float w2[12*npts+32];
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
	vector<pair<PhGUtils::Point3f, int>> targets;
	vector<float> w_landmarks;

	// fitting control
	static const int INITFRAMES = 5;
	int frameCounter;
	bool fitPose, fitIdentity, fitExpression;

	// fitting error
	float E;
};


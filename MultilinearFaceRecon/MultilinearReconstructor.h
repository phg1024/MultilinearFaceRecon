#pragma once

#include <QApplication>

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
#include "levmar.h"
#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include <armadillo>

class MultilinearReconstructor : public QObject
{
	Q_OBJECT
public:
	enum FittingOption {
		FIT_POSE,
		FIT_IDENTITY,
		FIT_ALL
	};

	MultilinearReconstructor(void);
	~MultilinearReconstructor(void);

	const Tensor1<float>& templateMesh() const {
		return tplt;
	}

	const Tensor1<float>& currentMesh() const {
		return tmesh;
	}

	// for reconstruction with 2D feature points, first obtain
	// corresponding 3D locations, then use the reconstruction
	// for 3D points

	// for reconstruction with 3D locations of feature points
	void bindTarget(const vector<pair<PhGUtils::Point3f, int>>& pts);
	void init();
	void fit(FittingOption ops = FIT_ALL);
	void fit_withPrior();

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
	void updateCoreC();
	void updateTMC();
	void transformTM0C();
	void transformTM1C();

private:
	friend void evalCost(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost2(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3(float *p, float *hx, int m, int n, void* adata);

	void transformMesh();
	float computeError();

	bool fitRigidTransformationOnly();
	bool fitRigidTransformationAndScale();
	bool fitIdentityWeights();
	bool fitIdentityWeights_withPrior();
	bool fitExpressionWeights();
	bool fitExpressionWeights_withPrior();

private:
	// convergence criteria
	float cc;
	float errorThreshold;
	static const int MAXITERS = 32;
	bool usePrior;

	// weights for prior
	float w_data, w_prior_id, w_prior_exp;
	float w_boundary;

	// the input core tensor
	Tensor3<float> core;
	// the truncated input core tensor
	Tensor3<float> corec;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;
		
	// the tensor after mode product, with truncation
	// tm0c: corec mode product with wid
	// tm1c: corec mode product with wexp
	Tensor2<float> tm0c, tm1c;
	// the tensor after 2 mode products, with truncation; 
	// tmc is the tensor before applying global transformation
	Tensor1<float> tmc;
	
	Tensor1<float> q;			// target point coordinates

	// template face
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;
	
	float RTparams[7]; /* sx, ry, rz, tx, ty, tz, scale */	
	// used to avoid local minima
	float meanX, meanY, meanZ;
	float scale;
	arma::fmat R;
	arma::fvec T;
	PhGUtils::Matrix3x3f Rmat;
	PhGUtils::Point3f Tvec;

	// computation related
	PhGUtils::DenseMatrix<float> Aid, Aexp;
	PhGUtils::DenseVector<float> brhs;

	// weights
	Tensor1<float> Wid, Wexp;

	// weights prior
	arma::fvec mu_wid, mu_wexp;
	arma::fmat sigma_wid, sigma_wexp;

	// target vertices	
	vector<pair<PhGUtils::Point3f, int>> targets;
	vector<float> w_landmarks;

	// fitting control
	static const int INITFRAMES = 5;
	int frameCounter;
	bool fitPose, fitIdentity, fitExpression;
};


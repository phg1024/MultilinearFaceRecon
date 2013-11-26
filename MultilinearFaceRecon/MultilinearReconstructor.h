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
using namespace arma;

class MultilinearReconstructor : public QObject
{
	Q_OBJECT
public:
	MultilinearReconstructor(void);
	~MultilinearReconstructor(void);

	const Tensor1<float>& templateMesh() const {
		return tplt;
	}

	const Tensor1<float>& currentMesh() const {
		return tmesh;
	}

	void bindTarget(const vector<pair<Point3f, int>>& pts);
	void init();
	void fit();

	const Tensor1<float>& expressionWeights() const { return Wexp; }
	const Tensor1<float>& identityWeights() const { return Wid; }

signals:
	void oneiter();

private:
	void loadTargetMesh();
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

	bool fitRigidTransformation();
	bool fitIdentityWeights();
	bool fitExpressionWeights();

private:
	// convergence criteria
	float cc;
	float errorThreshold;
	static const int MAXITERS = 128;

	// the input core tensor
	Tensor3<float> core;
	// the truncated input core tensor
	Tensor3<float> corec;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;
		
	// the tensor after mode product, with truncation
	Tensor2<float> tm0c, tm1c;
	// the tensor after 2 mode products, with truncation; 
	// tmc is the tensor before applying global transformation
	Tensor1<float> tmc;
	
	Tensor1<float> q;			// target point coordinates

	// template face
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;
	
	fmat R;
	fvec T;
	Matrix3x3f Rmat;
	Point3f Tvec;

	// computation related
	DenseMatrix<float> Aid, Aexp;
	DenseVector<float> brhs;

	// weights
	Tensor1<float> Wid, Wexp;

	// target vertices
	vector<int> landmarks;
	vector<pair<Point3f, int>> targets;
};


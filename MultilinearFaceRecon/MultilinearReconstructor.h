#pragma once

#include <QApplication>

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
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

	void fit();

signals:
	void oneiter();

private:
	void loadCoreTensor();
	void createTemplateItem();
	void unfoldTensor();
	void initializeWeights();

	void updateComputationTensor();
	void updateTM0C();
	void updateTM1C();
	void updateTMC();

private:
	friend void evalCost(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost2(float *p, float *hx, int m, int n, void* adata);
	friend void evalCost3(float *p, float *hx, int m, int n, void* adata);

	void transformMesh();
	float computeError();

	bool fitRigidTransformation(float cc = 1e-4);
	bool fitIdentityWeights(float cc = 1e-4);
	bool fitExpressionWeights(float cc = 1e-4);

private:
	// the input core tensor
	Tensor3<float> core;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;
		
	// the tensor after mode product, after truncation
	Tensor2<float> tm0c, tm1c;
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

	// weights
	Tensor1<float> Wid, Wexp;

	// target vertices
	vector<pair<Point3f, int>> targets;
};


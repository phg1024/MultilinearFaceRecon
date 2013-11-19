#pragma once

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
#include <armadillo>
using namespace arma;

class MultilinearReconstructor
{
public:
	MultilinearReconstructor(void);
	~MultilinearReconstructor(void);

	const Tensor1<float>& templateMesh() const {
		return tplt;
	}

	const Tensor1<float>& currentMesh() const {
		return tmesh;
	}

	void bindTarget(const vector<pair<Point3f, int>>& pts) {
		targets = pts;

		updateComputationTensor();
	}

	void fit();

private:
	void loadCoreTensor();
	void createTemplateItem();
	void unfoldTensor();
	void initializeWeights();

	void updateComputationTensor();

private:
	friend void evalCost(float *p, float *hx, int m, int n, void* adata);

	void transformMesh();

	void fitRigidTransformation();
	void fitIdentityWeights();
	void fitExpressionWeights();

private:
	// the input core tensor
	Tensor3<float> core;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// the tensor after mode product
	Tensor2<float> tm0, tm1;
		
	// the tensor after mode product, after truncation
	Tensor2<float> tm0c, tm1c;
	
	Tensor1<float> q;			// target point coordinates

	// template face
	Tensor1<float> tplt;

	// fitted face
	Tensor1<float> tmesh;
	
	fmat R;
	fvec T;

	// weights
	Tensor1<float> Wid, Wexp;

	// target vertices
	vector<pair<Point3f, int>> targets;
};


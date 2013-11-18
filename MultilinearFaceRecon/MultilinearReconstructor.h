#pragma once

#include "Math/Tensor.hpp"

class MultilinearReconstructor
{
public:
	MultilinearReconstructor(void);
	~MultilinearReconstructor(void);

	const Tensor1<float>& templateMesh() const {
		return tplt;
	}

private:
	void loadCoreTensor();
	void createTemplateItem();
	void unfoldTensor();
	void initializeWeights();

private:
	// the input core tensor
	Tensor3<float> core;

	// the unfolded tensor
	Tensor2<float> tu0, tu1;

	// template face
	Tensor1<float> tplt;

	// weights
	Tensor1<float> Wid, Wexp;
};


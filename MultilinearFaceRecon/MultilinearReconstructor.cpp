#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"

MultilinearReconstructor::MultilinearReconstructor(void)
{
	loadCoreTensor();
	unfoldTensor();
	createTemplateItem();

	initializeWeights();
}


MultilinearReconstructor::~MultilinearReconstructor(void)
{
}

void MultilinearReconstructor::loadCoreTensor()
{
	const string filename = "../Data/blendshape_core.tensor";
	core.read(filename);
}

void MultilinearReconstructor::unfoldTensor()
{
	tu0 = core.unfold(0);
	tu1 = core.unfold(1);
}

void MultilinearReconstructor::createTemplateItem()
{
	message("creating template face ...");
	// take the average of neutral face as the template
	tplt.resize(core.dim(2));

	for(int i=0;i<core.dim(0);i++) {
		for(int j=0;j<tplt.length();j++) {		
			tplt(j) += core(i, 0, j);
		}
	}

	double invCount = 1.0 / core.dim(0);

	for(int j=0;j<tplt.length();j++) {
		tplt(j) *= invCount;
	}
	message("done.");
}

void MultilinearReconstructor::initializeWeights()
{
	Wid.resize(core.dim(0));
	Wexp.resize(core.dim(1));

	float w0 = 1.0 / core.dim(0);
	for(int i=0;i<Wid.length();i++) {
		Wid(i) = w0;
	}

	// use neutral face initially
	for(int i=0;i<Wexp.length();i++) {
		Wexp(i) = 0;
	}
	Wexp(0) = 1.0;
}

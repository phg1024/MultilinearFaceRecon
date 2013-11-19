#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"

MultilinearReconstructor::MultilinearReconstructor(void)
{
	loadCoreTensor();
	unfoldTensor();
	initializeWeights();

	createTemplateItem();
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
	tplt = tm1.modeProduct(Wid, 0);

	tmesh = tplt;
	message("done.");
}

void MultilinearReconstructor::initializeWeights()
{
	message("initializing weights ...");
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

	tm0 = core.modeProduct(Wid, 0);
	tm1 = core.modeProduct(Wexp, 1);
	message("done.");
}

void MultilinearReconstructor::updateComputationTensor()
{
	int npts = targets.size();

	tm0c.resize(tm0.dim(0), npts * 3);

	for(int i=0;i<tm0.dim(0);i++) {
		for(int j=0, idx=0;j<npts;j++, idx+=3) {
			int vidx = targets[j].second * 3;
			tm0c(i, idx) = tm0(i, vidx);
			tm0c(i, idx+1) = tm0(i, vidx+1);
			tm0c(i, idx+2) = tm0(i, vidx+2);
		}
	}

	tm1c.resize(tm1.dim(0), npts * 3);
	for(int i=0;i<tm1.dim(0);i++) {
		for(int j=0, idx=0;j<npts;j++, idx+=3) {
			int vidx = targets[j].second * 3;
			tm1c(i, idx) = tm1(i, vidx);
			tm1c(i, idx+1) = tm1(i, vidx+1);
			tm1c(i, idx+2) = tm1(i, vidx+2);
		}
	}

	q.resize(npts*3);

	for(int i=0, idx=0;i<npts;i++, idx+=3) {
		int vidx = targets[i].second;
		const Point3f& p = targets[i].first;	
		q(idx) = p.x;
		q(idx+1) = p.y;
		q(idx+2) = p.z;
	}
}

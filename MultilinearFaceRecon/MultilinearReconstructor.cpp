#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"
#include "levmar.h"

MultilinearReconstructor::MultilinearReconstructor(void)
{
	loadCoreTensor();
	unfoldTensor();
	initializeWeights();
	createTemplateItem();

	R = fmat(3, 3);
	T = fvec(3);
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

void MultilinearReconstructor::fit()
{
	fitRigidTransformation();
	fitIdentityWeights();
	fitExpressionWeights();	
}

void evalCost(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmesh = recon->tmesh;

	// set up rotation matrix and translation vector
	Point3f T(tx, ty, tz);
	Matrix3x3f R = rotationMatrix(rx, ry, rz) * s;

	float cost = 0;
	for(int i=0;i<npts;i++) {
		int vidx = targets[i].second * 3;
		Point3f p(tmesh(vidx),	tmesh(vidx+1), tmesh(vidx+2));
		const Point3f& q = targets[i].first;

		Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

void MultilinearReconstructor::fitRigidTransformation()
{
	float params[7] = {1.0, 0, 0, 0, 0, 0, 0};		/* scale, rx, ry, rz, tx, ty, tz */	
	int npts = targets.size();
	vector<float> meas(npts);
	int iters = slevmar_dif(evalCost, params, &meas[0], 7, npts, 128, NULL, NULL, NULL, NULL, this);
	cout << "finished in " << iters << " iterations." << endl;

	// set up the matrix and translation vector
	Matrix3x3f Rmat = rotationMatrix(params[1], params[2], params[3]) * params[0];
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			R(i, j) = Rmat(i, j);
		}
	}
	T(0) = params[4], T(1) = params[5], T(2) = params[6];
	cout << R << endl;
	cout << T << endl;

	transformMesh();
}

void MultilinearReconstructor::fitIdentityWeights()
{

}

void MultilinearReconstructor::fitExpressionWeights()
{

}

void MultilinearReconstructor::transformMesh()
{
	int nverts = tmesh.length()/3;
	fmat pt(3, nverts);
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		pt(0, i) = tmesh(idx);
		pt(1, i) = tmesh(idx+1);
		pt(2, i) = tmesh(idx+2);
	}

	fmat pt_trans =  R * pt;
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		tmesh(idx) = pt_trans(0, i) + T(0);
		tmesh(idx+1) = pt_trans(1, i) + T(1);
		tmesh(idx+2) = pt_trans(2, i) + T(2);
	}
}
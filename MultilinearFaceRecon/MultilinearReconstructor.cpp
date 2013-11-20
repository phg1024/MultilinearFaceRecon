#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"
#include "Math/denseblas.h"
#define USELEVMAR4WEIGHTS 0

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
	updateTM0C();
	updateTM1C();
	updateTMC();
}

void MultilinearReconstructor::updateTM0C() {
	// update tm0c
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
}

void MultilinearReconstructor::updateTM1C() {
	// update tm1c
	int npts = targets.size();

	tm1c.resize(tm1.dim(0), npts * 3);
	for(int i=0;i<tm1.dim(0);i++) {
		for(int j=0, idx=0;j<npts;j++, idx+=3) {
			int vidx = targets[j].second * 3;
			tm1c(i, idx) = tm1(i, vidx);
			tm1c(i, idx+1) = tm1(i, vidx+1);
			tm1c(i, idx+2) = tm1(i, vidx+2);
		}
	}
}

void MultilinearReconstructor::updateTMC() {
	// update tmc
	int npts = targets.size();

	tmc.resize(npts * 3);
	for(int i=0;i<tmc.length();i++) {
		float val = 0;
		for(int j=0;j<tm1c.dim(0);j++) {
			val += tm1c(j, i) * Wid(j);
		}
		tmc(i) = val;
	}
}

void MultilinearReconstructor::fit()
{
	const int MAXITERS = 64;
	int iters = 0;
	bool converged = false;
	while( !converged && iters++ < MAXITERS ) {
		//converged = true;		
		fitRigidTransformation();
		fitIdentityWeights();
		fitExpressionWeights();	
		tplt = tm1.modeProduct(Wid, 0);
		transformMesh();

		converged = computeError() < 1e-3;
		emit oneiter();
		QApplication::processEvents();

		//system("pause");
	}
	cout << "Total iterations = " << iters << endl;
}

void evalCost(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	// set up rotation matrix and translation vector
	Point3f T(tx, ty, tz);
	Matrix3x3f R = rotationMatrix(rx, ry, rz) * s;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		const Point3f& q = targets[i].first;

		Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

void MultilinearReconstructor::fitRigidTransformation(float cc)
{
	float params[7] = {1.0, 0, 0, 0, 0, 0, 0};		/* scale, rx, ry, rz, tx, ty, tz */	
	updateTMC();

	int npts = targets.size();
	vector<float> meas(npts);
	int iters = slevmar_dif(evalCost, params, &(meas[0]), 7, npts, 128, NULL, NULL, NULL, NULL, this);
	cout << "finished in " << iters << " iterations." << endl;

	// set up the matrix and translation vector
	Rmat = rotationMatrix(params[1], params[2], params[3]) * params[0];

	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			R(i, j) = Rmat(i, j);
		}
	}

	Tvec = Point3f(params[4], params[5], params[6]);	
	T(0) = params[4], T(1) = params[5], T(2) = params[6];
	//cout << R << endl;
	//cout << T << endl;
}

void evalCost2(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tm1c = recon->tm1c;

	// set up rotation matrix and translation vector
	const Point3f& T = recon->Tvec;
	const Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm1c(j, vidx) * p[j];
			y += tm1c(j, vidx+1) * p[j];
			z += tm1c(j, vidx+2) * p[j];
		}
		Point3f p(x, y, z);
		const Point3f& q = targets[i].first;

		Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

void MultilinearReconstructor::fitIdentityWeights(float cc)
{
#if USELEVMAR4WEIGHTS
	int nparams = core.dim(0);
	vector<float> params(nparams);
	int npts = targets.size();
	vector<float> meas(npts);
	int iters = slevmar_dif(evalCost2, &(params[0]), &(meas[0]), nparams, npts, 1024, NULL, NULL, NULL, NULL, this);
	cout << "finished in " << iters << " iterations." << endl;

	for(int i=0;i<nparams;i++) {
		Wid(i) = params[i];		
		//cout << params[i] << ' ';
	}
	//cout << endl;
#else
	// to use this method, the tensor tm1c must first be updated using the rotation matrix and translation vector
	int nparams = core.dim(0);
	vector<float> params(nparams);
	// assemble the matrix
	DenseMatrix<float> A(tm1c.dim(1), tm1c.dim(0));
	DenseVector<float> b(q.length());

	for(int i=0;i<tm1c.dim(0);i++) {
		for(int j=0;j<tm1c.dim(1);j++) {
			A(j, i) = tm1c(i, j);
		}
	}

	for(int i=0;i<q.length();i++) {
		b(i) = q(i);
	}

	int rtn = leastsquare<float>(A, b);
	//debug("rtn", rtn);

	//b.print("b");
	for(int i=0;i<nparams;i++) {
		Wid(i) = b(i);		
		//cout << params[i] << ' ';
	}
	//cout << endl;
#endif

	// update the mesh with identity weights
	//tplt = tm1.modeProduct(Wid, 0);
	//tmesh = tplt;
	//transformMesh();

	// also update the tensor after mode product
	tm0 = core.modeProduct(Wid, 0);

	updateTM0C();
}

void evalCost3(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tm0c = recon->tm0c;

	// set up rotation matrix and translation vector
	const Point3f& T = recon->Tvec;
	const Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm0c(j, vidx) * p[j];
			y += tm0c(j, vidx+1) * p[j];
			z += tm0c(j, vidx+2) * p[j];
		}
		Point3f p(x, y, z);
		const Point3f& q = targets[i].first;

		Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

void MultilinearReconstructor::fitExpressionWeights(float cc)
{
#if USELEVMAR4WEIGHTS
	// fix both rotation and identity weights, solve for expression weights
	int nparams = core.dim(1);
	vector<float> params(nparams);
	int npts = targets.size();
	vector<float> meas(npts);
	int iters = slevmar_dif(evalCost3, &(params[0]), &(meas[0]), nparams, npts, 1024, NULL, NULL, NULL, NULL, this);

	cout << "finished in " << iters << " iterations." << endl;

	for(int i=0;i<nparams;i++) {
		Wexp(i) = params[i];
		//cout << params[i] << ' ';
	}
	//cout << endl;
#else
	int nparams = core.dim(1);
	DenseMatrix<float> A(tm0c.dim(1), tm0c.dim(0));
	DenseVector<float> b(q.length());
	for(int i=0;i<tm0c.dim(0);i++) {
		for(int j=0;j<tm0c.dim(1);j++) {
			A(j, i) = tm0c(i, j);
		}
	}
	
	for(int i=0;i<q.length();i++) {
		b(i) = q(i);
	}

	int rtn = leastsquare<float>(A, b);
	//debug("rtn", rtn);

	//b.print("b");
	for(int i=0;i<nparams;i++) {
		Wexp(i) = b(i);
		//cout << params[i] << ' ';
	}
	//cout << endl;
	//cout << endl;
#endif
	// update the mesh with identity weights
	//tplt = tm0.modeProduct(Wexp, 0);
	//tmesh = tplt;

	// also update the tensor after mode product
	tm1 = core.modeProduct(Wexp, 1);
	updateTM1C();
}

void MultilinearReconstructor::transformMesh()
{
	int nverts = tplt.length()/3;
	fmat pt(3, nverts);
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		pt(0, i) = tplt(idx);
		pt(1, i) = tplt(idx+1);
		pt(2, i) = tplt(idx+2);
	}

	fmat pt_trans =  R * pt;
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		tmesh(idx) = pt_trans(0, i) + T(0);
		tmesh(idx+1) = pt_trans(1, i) + T(1);
		tmesh(idx+2) = pt_trans(2, i) + T(2);
	}
}

void MultilinearReconstructor::bindTarget( const vector<pair<Point3f, int>>& pts )
{
	targets = pts;
	int npts = targets.size();

	// initialize q
	q.resize(npts*3);	
	for(int i=0, idx=0;i<npts;i++, idx+=3) {
		int vidx = targets[i].second;
		const Point3f& p = targets[i].first;	
		q(idx) = p.x;
		q(idx+1) = p.y;
		q(idx+2) = p.z;
	}

	updateComputationTensor();
}

float MultilinearReconstructor::computeError()
{
	int npts = targets.size();
	float E = 0;
	for(int i=0;i<npts;i++) {
		int vidx = targets[i].second * 3;
		Point3f p(tmesh(vidx), tmesh(vidx+1), tmesh(vidx+2));
		E += p.squaredDistanceTo(targets[i].first);
	}

	debug("Error", E);
	return E;
}

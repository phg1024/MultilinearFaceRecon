#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"
#include "Math/denseblas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#define USELEVMAR4WEIGHTS 0

MultilinearReconstructor::MultilinearReconstructor(void)
{
	loadCoreTensor();
	loadPrior();
	init();

	cc = 1e-6;
	errorThreshold = 1e-3;
}

MultilinearReconstructor::~MultilinearReconstructor(void)
{
}


void MultilinearReconstructor::loadPrior()
{
	// the prior data is stored in the following format
	// the matrices are stored in column major order
	
	// ndims
	// mean vector
	// covariance matrix
	
	cout << "loading prior data ..." << endl;
	const string& fnwid  = "../Data/wid.bin";
	ifstream fwid(fnwid, ios::in | ios::binary );

	int ndims;
	fwid.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	cout << "identity prior dim = " << ndims << endl;

	mu_wid.resize(ndims);
	sigma_wid.resize(ndims, ndims);

	fwid.read(reinterpret_cast<char*>(mu_wid.memptr()), sizeof(float)*ndims);
	fwid.read(reinterpret_cast<char*>(sigma_wid.memptr()), sizeof(float)*ndims*ndims);

	fwid.close();

	mu_wid.print("mean_wid");
	sigma_wid.print("sigma_wid");

	const string& fnwexp = "../Data/wexp.bin";
	ifstream fwexp(fnwexp, ios::in | ios::binary );

	fwexp.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	cout << "expression prior dim = " << ndims << endl;

	mu_wexp.resize(ndims);
	sigma_wexp.resize(ndims, ndims);

	fwexp.read(reinterpret_cast<char*>(mu_wexp.memptr()), sizeof(float)*ndims);
	fwexp.read(reinterpret_cast<char*>(sigma_wexp.memptr()), sizeof(float)*ndims*ndims);

	fwexp.close();

	mu_wexp.print("mean_wexp");
	sigma_wexp.print("sigma_wexp");
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

void MultilinearReconstructor::init()
{
	initializeWeights();
	createTemplateItem();
	
	R = fmat(3, 3);
	R(0, 0) = 1.0, R(1, 1) = 1.0, R(2, 2) = 1.0;
	T = fvec(3);

	Rmat = Matrix3x3f::identity();
	Tvec = Point3f::zero();

	updateComputationTensor();
}

void MultilinearReconstructor::fit()
{
	init();

	if(targets.empty())
	{
		error("No target set!");
		return;
	}
	int iters = 0;
	float E0 = 0;
	bool converged = false;
	while( !converged && iters++ < MAXITERS ) {
		converged = true;
		converged &= fitRigidTransformation();

		// apply the new global transformation to tm1c
		// because tm1c is required in fitting identity weights
		transformTM1C();

		converged &= fitIdentityWeights();
		// update tm0c with the new identity weights
		// now the tensor is not updated with global rigid transformation
		tm0c = corec.modeProduct(Wid, 0);
		// apply the global transformation to tm0c
		// because tm0c is required in fitting expression weights
		transformTM0C();

		converged &= fitExpressionWeights();	
		// update tm1c with the new expression weights
		// now the tensor is not updated with global rigid transformation
		tm1c = corec.modeProduct(Wexp, 1);
		// compute tmc from the new tm1c
		updateTMC();

		// uncomment to show the transformation process
		//transformMesh();

		float E = computeError();
		debug("iters", iters, "Error", E);

		converged |= E < errorThreshold;		
		E0 = E;
		//emit oneiter();
		//QApplication::processEvents();
	}
	cout << "Total iterations = " << iters << endl;
	transformMesh();
	//emit oneiter();
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

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		const Point3f& q = targets[i].first;

		Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

bool MultilinearReconstructor::fitRigidTransformation()
{
	float params[7] = {1.0, 0, 0, 0, 0, 0, 0};		/* scale, rx, ry, rz, tx, ty, tz */	

	int npts = targets.size();
	vector<float> meas(npts);
	int iters = slevmar_dif(evalCost, params, &(meas[0]), 7, npts, 128, NULL, NULL, NULL, NULL, this);
	//cout << "finished in " << iters << " iterations." << endl;

	// set up the matrix and translation vector
	Rmat = rotationMatrix(params[1], params[2], params[3]) * params[0];
	float diff = 0;
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			diff += fabs(R(i, j) - Rmat(i, j));
			R(i, j) = Rmat(i, j);			
		}
	}

	Tvec = Point3f(params[4], params[5], params[6]);	
	diff += fabs(Tvec.x - T(0)) + fabs(Tvec.y - T(1)) + fabs(Tvec.z - T(2));
	T(0) = params[4], T(1) = params[5], T(2) = params[6];

	//cout << R << endl;
	//cout << T << endl;
	return diff / 7 < cc;
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

bool MultilinearReconstructor::fitIdentityWeights()
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

	// assemble the matrix
	for(int i=0;i<tm1c.dim(0);i++) {
		for(int j=0;j<tm1c.dim(1);j++) {
			Aid(j, i) = tm1c(i, j);
		}
	}

	for(int i=0;i<q.length();i++) {
		brhs(i) = q(i);
	}

	int rtn = leastsquare<float>(Aid, brhs);
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wid(i) - brhs(i));
		Wid(i) = brhs(i);		
		//cout << params[i] << ' ';
	}
	//cout << endl;
#endif

	return diff / nparams < cc;
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

bool MultilinearReconstructor::fitExpressionWeights()
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

	for(int i=0;i<tm0c.dim(0);i++) {
		for(int j=0;j<tm0c.dim(1);j++) {
			Aexp(j, i) = tm0c(i, j);
		}
	}
	
	for(int i=0;i<q.length();i++) {
		brhs(i) = q(i);
	}

	int rtn = leastsquare<float>(Aexp, brhs);
	//debug("rtn", rtn);

	//b.print("b");
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff += abs(Wexp(i) - brhs(i));
		Wexp(i) = brhs(i);
		//cout << params[i] << ' ';
	}
	//cout << endl;
	//cout << endl;
#endif

	return diff / nparams < cc;
}

void MultilinearReconstructor::transformMesh()
{
	tplt = core.modeProduct(Wexp, 1).modeProduct(Wid, 0);

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
}

void MultilinearReconstructor::updateComputationTensor()
{
	updateCoreC();
	tm0c = corec.modeProduct(Wid, 0);
	tm1c = corec.modeProduct(Wexp, 1);
	updateTMC();

	Aid = DenseMatrix<float>::zeros(tm1c.dim(1), tm1c.dim(0));
	Aexp = DenseMatrix<float>::zeros(tm0c.dim(1), tm0c.dim(0));
	brhs.resize(targets.size()*3);
}

// build a truncated version of the core
void MultilinearReconstructor::updateCoreC() {
	corec.resize(core.dim(0), core.dim(1), targets.size()*3);

	for(int i=0;i<core.dim(0);i++) {
		for(int j=0;j<core.dim(1);j++) {
			for(int k=0, idx=0;k<targets.size();k++, idx+=3) {
				int vidx = targets[k].second * 3;
				corec(i, j, idx) = core(i, j, vidx);
				corec(i, j, idx+1) = core(i, j, vidx+1);
				corec(i, j, idx+2) = core(i, j, vidx+2);
			}
		}
	}
}

// transform TM0C with global rigid transformation
void MultilinearReconstructor::transformTM0C() {
	int npts = tm0c.dim(1) / 3;
	for(int i=0;i<tm0c.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {
			Point3f p(tm0c(i, vidx), tm0c(i, vidx+1), tm0c(i, vidx+2));
			p = Rmat * p + Tvec;
			tm0c(i, vidx) = p.x;
			tm0c(i, vidx+1) = p.y;
			tm0c(i, vidx+2) = p.z;
		}
	}
}

// transform TM1C with global rigid transformation
void MultilinearReconstructor::transformTM1C() {
	int npts = tm1c.dim(1) / 3;
	for(int i=0;i<tm1c.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {
			Point3f p(tm1c(i, vidx), tm1c(i, vidx+1), tm1c(i, vidx+2));
			p = Rmat * p + Tvec;
			tm1c(i, vidx) = p.x;
			tm1c(i, vidx+1) = p.y;
			tm1c(i, vidx+2) = p.z;
		}
	}
}

void MultilinearReconstructor::updateTMC() {
	tmc = tm1c.modeProduct(Wid, 0);
}

float MultilinearReconstructor::computeError()
{
	int npts = targets.size();
	float E = 0;
	for(int i=0;i<npts;i++) {
		int vidx = i * 3;
		Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		p = Rmat * p + Tvec;
		E += p.squaredDistanceTo(targets[i].first);
	}
	return E;
}
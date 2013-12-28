#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"
#include "Math/denseblas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#define USELEVMAR4WEIGHTS 0
#define USE_MKL_LS 1		// use mkl least square solver
#define OUTPUT_STATS 0

MultilinearReconstructor::MultilinearReconstructor(void)
{	
	w_prior_id = 5e-4;
	// for 25 expression dimensions
	//w_prior_exp = 1e-4;
	// for 47 expression dimensions
	w_prior_exp = 5e-4;
	w_boundary = 1e-8;

	meanX = meanY = meanZ = 0;
	
	mkl_set_num_threads(8);

	loadCoreTensor();
	loadPrior();
	initializeWeights();
	createTemplateItem();
	//updateComputationTensor();
	init();

	w_landmarks.resize(512);

	cc = 1e-4;
	errorThreshold = 5e-5;
	errorDiffThreshold = errorThreshold * 0.01;
	usePrior = true;

	frameCounter = 0;

	useHistory = true;
	historyWeights[0] = 0.02;
	historyWeights[1] = 0.04;
	historyWeights[2] = 0.07;
	historyWeights[3] = 0.10;
	historyWeights[4] = 0.15;
}

MultilinearReconstructor::~MultilinearReconstructor(void)
{
	culaShutdown();
}


void MultilinearReconstructor::togglePrior()
{
	usePrior = !usePrior;
	PhGUtils::message((usePrior)?"Using prior":"Not using prior");
	init();
}


void MultilinearReconstructor::loadPrior()
{
	// the prior data is stored in the following format
	// the matrices are stored in column major order

	// ndims
	// mean vector
	// covariance matrix

	cout << "loading prior data ..." << endl;
	const string fnwid  = "../Data/blendshape/wid.bin";
	ifstream fwid(fnwid, ios::in | ios::binary );

	int ndims;
	fwid.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	cout << "identity prior dim = " << ndims << endl;

	mu_wid0.resize(ndims);
	mu_wid.resize(ndims);
	sigma_wid.resize(ndims, ndims);

	fwid.read(reinterpret_cast<char*>(mu_wid0.memptr()), sizeof(float)*ndims);
	fwid.read(reinterpret_cast<char*>(mu_wid.memptr()), sizeof(float)*ndims);
	fwid.read(reinterpret_cast<char*>(sigma_wid.memptr()), sizeof(float)*ndims*ndims);

	fwid.close();

	PhGUtils::message("identity prior loaded.");
	PhGUtils::message("processing identity prior.");
	//mu_wid.print("mean_wid");
	//sigma_wid.print("sigma_wid");
	sigma_wid = arma::inv(sigma_wid);
	sigma_wid_weighted = sigma_wid * w_prior_id;
	mu_wid = sigma_wid * mu_wid;
	mu_wid_weighted = mu_wid * w_prior_id;
	PhGUtils::message("done");

	const string fnwexp = "../Data/blendshape/wexp.bin";
	ifstream fwexp(fnwexp, ios::in | ios::binary );

	fwexp.read(reinterpret_cast<char*>(&ndims), sizeof(int));
	cout << "expression prior dim = " << ndims << endl;

	mu_wexp0.resize(ndims);
	mu_wexp.resize(ndims);
	sigma_wexp.resize(ndims, ndims);

	fwexp.read(reinterpret_cast<char*>(mu_wexp0.memptr()), sizeof(float)*ndims);
	fwexp.read(reinterpret_cast<char*>(mu_wexp.memptr()), sizeof(float)*ndims);
	fwexp.read(reinterpret_cast<char*>(sigma_wexp.memptr()), sizeof(float)*ndims*ndims);

	fwexp.close();
	//mu_wexp.print("mean_wexp");
	//sigma_wexp.print("sigma_wexp");
	sigma_wexp = arma::inv(sigma_wexp);
	sigma_wexp_weighted = sigma_wexp * w_prior_exp;
	mu_wexp = sigma_wexp * mu_wexp;
	mu_wexp_weighted = mu_wexp * w_prior_exp;
}

void MultilinearReconstructor::loadCoreTensor()
{
	const string filename = "../Data/blendshape/core.bin";
	core.read(filename);
}

void MultilinearReconstructor::unfoldTensor()
{
	tu0 = core.unfold(0);
	tu1 = core.unfold(1);
}

void MultilinearReconstructor::createTemplateItem()
{
	PhGUtils::message("creating template face ...");
	tplt = tm1.modeProduct(Wid, 0);
	tmesh = tplt;
	PhGUtils::message("done.");
}

void MultilinearReconstructor::initializeWeights()
{
	PhGUtils::message("initializing weights ...");
	Wid.resize(core.dim(0));
	Wexp.resize(core.dim(1));

	// use the first person initially
	for(int i=0;i<Wid.length();i++) {
		Wid(i) = mu_wid0(i);
	}

	// use neutral face initially
	for(int i=0;i<Wexp.length();i++) {
		Wexp(i) = mu_wexp0(i);
	}

	tm0 = core.modeProduct(Wid, 0);
	tm1 = core.modeProduct(Wexp, 1);
	PhGUtils::message("done.");
}

void MultilinearReconstructor::init()
{
	//initializeWeights();
	//createTemplateItem();

	RTparams[0] = 0; RTparams[1] = 0; RTparams[2] = 0;
	RTparams[3] = meanX; RTparams[4] = meanY; RTparams[5] = meanZ;
	RTparams[6] = 1.0;

	R = arma::fmat(3, 3);
	R(0, 0) = 1.0, R(1, 1) = 1.0, R(2, 2) = 1.0;
	T = arma::fvec(3);

	Rmat = PhGUtils::Matrix3x3f::identity();
	Tvec = PhGUtils::Point3f::zero();

	//updateComputationTensor();

	updateMatrices();
}

void MultilinearReconstructor::fit2d(FittingOption ops /*= FIT_ALL*/)
{
	switch( ops ) {
	case FIT_POSE:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = false;
			break;
		}
	case FIT_IDENTITY:
		{
			fitPose = false;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_EXPRESSION:
		{
			fitPose = false;
			fitIdentity = false;
			fitExpression = true;
			break;
		}
	case FIT_POSE_AND_IDENTITY:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = true;
			break;
		}
	case FIT_ALL:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = true;
			break;
		}
	}
	frameCounter++;

	if( usePrior ) {
		fit2d_withPrior();

		if( ops == FIT_POSE_AND_IDENTITY ) {
			updateTM0();
		}
		if( ops == FIT_ALL ) {
			updateTM0();
			updateTM1();
		}

		return;
	}
	else {
		float w_prior_exp_tmp = w_prior_exp;
		float w_prior_id_tmp = w_prior_id;

		// set the prior weights to 0
		w_prior_exp = 0;
		w_prior_id = 0;

		fit2d_withPrior();

		if( ops == FIT_POSE_AND_IDENTITY ) {
			updateTM0();
		}
		if( ops == FIT_ALL ) {
			updateTM0();
			updateTM1();
		}

		// restore the priors
		w_prior_exp = w_prior_exp_tmp;
		w_prior_id = w_prior_id_tmp;

		return;
	}
}

void MultilinearReconstructor::fit2d_withPrior() {
	//cout << "fit with prior" << endl;

	//init();

	if(targets.empty())
	{
		PhGUtils::error("No target set!");
		return;
	}
	PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

	timerTotal.tic();
	int iters = 0;
	float E0 = 0;
	bool converged = false;
	while( !converged && iters++ < MAXITERS ) {
		converged = true;

		if( fitPose ) {
			timerRT.tic();
			converged &= fitRigidTransformationAndScale_2D();		
			timerRT.toc();
		}

		if( fitIdentity ) {			
			// apply the new global transformation to tm1c
			// because tm1c is required in fitting identity weights
			timerOther.tic();
			transformTM1C();
			timerOther.toc();

			timerID.tic();
			converged &= fitIdentityWeights_withPrior_2D();
			timerID.toc();
		}

		if( fitIdentity && (fitExpression || fitPose) ) {
			timerOther.tic();
			// update tm0c with the new identity weights
			// now the tensor is not updated with global rigid transformation
			tm0c = corec.modeProduct(Wid, 0);
			//corec.modeProduct(Wid, 0, tm0c);
			timerOther.toc();
		}

		if( fitExpression ) {
			timerOther.tic();
			// apply the global transformation to tm0c
			// because tm0c is required in fitting expression weights
			transformTM0C();
			timerOther.toc();

			timerExp.tic();
			converged &= fitExpressionWeights_withPrior_2D();	
			timerExp.toc();
		}

		// this is not exactly logically correct
		// but this works for the case of fitting both pose and expression
		if( fitExpression && fitIdentity ) {//(fitIdentity || fitPose) ) {
			timerOther.tic();
			// update tm1c with the new expression weights
			// now the tensor is not updated with global rigid transformation
			tm1c = corec.modeProduct(Wexp, 1);
			//corec.modeProduct(Wexp, 1, tm1c);
			timerOther.toc();
		}
				
		// compute tmc from the new tm1c or new tm0c
		if( fitIdentity ) {
			timerOther.tic();
			//updateTMCwithTM0C();
			updateTMC();
			timerOther.toc();
		}
		else if( fitExpression ) {
			timerOther.tic();
			updateTMCwithTM0C();
			//updateTMC();
			timerOther.toc();
		}		

		timerOther.tic();

		float E = computeError_2D();
		//PhGUtils::debug("iters", iters, "Error", E);

		converged |= E < errorThreshold;
		converged |= fabs(E - E0) < errorDiffThreshold;
		E0 = E;
		timerOther.toc();

		// uncomment to show the transformation process		
		/*
		transformMesh();
		Rmat.print("R");
		Tvec.print("T");
		emit oneiter();
		QApplication::processEvents();

		::system("pause");
		*/
	}

	timerTransform.tic();
	if( fitIdentity && !fitExpression ) { 
		//cout << tm1.dim(0) << ", " << tm1.dim(1) << endl;
		//PhGUtils::debug("Wid", Wid);
		tplt = tm1.modeProduct(Wid, 0);
	}
	else if( fitExpression && !fitIdentity ) {
		//cout << tm0.dim(0) << ", " << tm0.dim(1) << endl;
		//PhGUtils::debug("Wexp", Wexp);
		tplt = tm0.modeProduct(Wexp, 0);
	}
	else if( fitIdentity && fitExpression )
		tplt = core.modeProduct(Wexp, 1).modeProduct(Wid, 0);
	timerTransform.toc();

	if( useHistory ) {
		// post process, impose a moving average for pose
		RTHistory.push_back(vector<float>(RTparams, RTparams+7));
		if( RTHistory.size() > historyLength ) RTHistory.pop_front();
		vector<float> meanRT = computeWeightedMeanPose();
		// copy back the mean pose
		for(int i=0;i<7;i++) RTparams[i] = meanRT[i];
	}

	timerTransform.tic();
	//PhGUtils::debug("R", Rmat);
	//PhGUtils::debug("T", Tvec);
	transformMesh();
	timerTransform.toc();
	//emit oneiter();
	timerTotal.toc();

#if OUTPUT_STATS
	cout << "Total iterations = " << iters << endl;
	PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed()*1000) + " ms.");
	PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed()*1000) + " ms.");
#endif
}

void MultilinearReconstructor::fit( MultilinearReconstructor::FittingOption ops )
{
	switch( ops ) {
	case FIT_POSE:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = false;
			break;
		}
	case FIT_IDENTITY:
		{
			fitPose = false;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_EXPRESSION:
		{
			fitPose = false;
			fitIdentity = false;
			fitExpression = true;
			break;
		}
	case FIT_POSE_AND_IDENTITY:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = true;
			break;
		}
	case FIT_ALL:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = true;
			break;
		}
	}
	frameCounter++;

	if( usePrior ) {
		fit_withPrior();

		if( ops == FIT_POSE_AND_IDENTITY ) {
			updateTM0();
		}
		if( ops == FIT_ALL ) {
			updateTM0();
			updateTM1();
		}

		return;
	}
	else {
		float w_prior_exp_tmp = w_prior_exp;
		float w_prior_id_tmp = w_prior_id;

		// set the prior weights to 0
		w_prior_exp = 0;
		w_prior_id = 0;

		fit_withPrior();

		if( ops == FIT_POSE_AND_IDENTITY ) {
			updateTM0();
		}
		if( ops == FIT_ALL ) {
			updateTM0();
			updateTM1();
		}

		// restore the priors
		w_prior_exp = w_prior_exp_tmp;
		w_prior_id = w_prior_id_tmp;

		return;
	}
}

void MultilinearReconstructor::fit_withPrior() {
	//cout << "fit with prior" << endl;

	//init();

	if(targets.empty())
	{
		PhGUtils::error("No target set!");
		return;
	}
	PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

	timerTotal.tic();
	int iters = 0;
	float E0 = 0;
	bool converged = false;
	while( !converged && iters++ < MAXITERS ) {
		converged = true;

		if( fitPose ) {
			timerRT.tic();
			converged &= fitRigidTransformationAndScale();		
			timerRT.toc();
		}

		if( fitIdentity ) {			
			// apply the new global rotation to tm1c
			// because tm1c is required in fitting identity weights
			timerOther.tic();
			transformTM1C();
			timerOther.toc();

			timerID.tic();
			converged &= fitIdentityWeights_withPrior();
			timerID.toc();
		}

		if( fitIdentity && (fitExpression || fitPose) ) {
			timerOther.tic();
			// update tm0c with the new identity weights
			// now the tensor is not updated with global rigid transformation
			tm0c = corec.modeProduct(Wid, 0);
			//corec.modeProduct(Wid, 0, tm0c);
			timerOther.toc();
		}

		if( fitExpression ) {
			timerOther.tic();
			// apply the global rotation to tm0c
			// because tm0c is required in fitting expression weights
			transformTM0C();
			timerOther.toc();

			timerExp.tic();
			converged &= fitExpressionWeights_withPrior();	
			timerExp.toc();
		}

		// this is not exactly logically correct
		// but this works for the case of fitting both pose and expression
		if( fitExpression && fitIdentity ) {//(fitIdentity || fitPose) ) {
			timerOther.tic();
			// update tm1c with the new expression weights
			// now the tensor is not updated with global rigid transformation
			tm1c = corec.modeProduct(Wexp, 1);
			//corec.modeProduct(Wexp, 1, tm1c);
			timerOther.toc();
		}

		//::system("pause");


		// compute tmc from the new tm1c or new tm0c
		if( fitIdentity ) {
			timerOther.tic();
			//updateTMCwithTM0C();
			updateTMC();
			timerOther.toc();
		}
		else if( fitExpression ) {
			timerOther.tic();
			updateTMCwithTM0C();
			//updateTMC();
			timerOther.toc();
		}		

		timerOther.tic();
		// uncomment to show the transformation process
		//transformMesh();
		//Rmat.print("R");
		//Tvec.print("T");
		float E = computeError();
		//PhGUtils::debug("iters", iters, "Error", E);

		converged |= E < errorThreshold;
		converged |= fabs(E - E0) < errorDiffThreshold;
		E0 = E;
		timerOther.toc();
		//emit oneiter();
		//QApplication::processEvents();
	}

	timerTransform.tic();
	if( fitIdentity && !fitExpression ) { 
		//cout << tm1.dim(0) << ", " << tm1.dim(1) << endl;
		//PhGUtils::debug("Wid", Wid);
		tplt = tm1.modeProduct(Wid, 0);
	}
	else if( fitExpression && !fitIdentity ) {
		//cout << tm0.dim(0) << ", " << tm0.dim(1) << endl;
		//PhGUtils::debug("Wexp", Wexp);
		tplt = tm0.modeProduct(Wexp, 0);
	}
	else if( fitIdentity && fitExpression )
		tplt = core.modeProduct(Wexp, 1).modeProduct(Wid, 0);
	timerTransform.toc();

	if( useHistory ) {
		// post process, impose a moving average for pose
		RTHistory.push_back(vector<float>(RTparams, RTparams+7));
		if( RTHistory.size() > historyLength ) RTHistory.pop_front();
		vector<float> meanRT = computeWeightedMeanPose();
		// copy back the mean pose
		for(int i=0;i<7;i++) RTparams[i] = meanRT[i];
	}

	timerTransform.tic();
	//PhGUtils::debug("R", Rmat);
	//PhGUtils::debug("T", Tvec);
	transformMesh();
	timerTransform.toc();
	//emit oneiter();
	timerTotal.toc();

#if OUTPUT_STATS
	cout << "Total iterations = " << iters << endl;
	PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed()*1000) + " ms.");
	PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed()*1000) + " ms.");
	PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed()*1000) + " ms.");
#endif
}

vector<float> MultilinearReconstructor::computeWeightedMeanPose() {
	vector<float> m(7, 0);

	float wsum = 0;
	int i=0;
	for(auto it=RTHistory.begin(); it!= RTHistory.end(); ++it) {
		for(int j=0;j<7;j++) {
			m[j] += (*it)[j] * historyWeights[i];
		}
		wsum += historyWeights[i];
		i++;
	}

	for(int j=0;j<7;j++) m[j] /= wsum;

	return m;
}

void evalCost_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++) {
		//PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		float wpt = w_landmarks[vidx];
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		const PhGUtils::Point3f& q = targets[i].first;

		// PhGUtils::Point3f pp = R * p + T;
		PhGUtils::transformPoint( px, py, pz, R, T );

		// Projection to image plane
		float u, v, d;
		PhGUtils::worldToColor( px, py, pz, u, v, d );

		/*
		// Back-Projection to world space
		float qx, qy, qz;
		PhGUtils::colorToWorld(q.x, q.y, q.z, qx, qy, qz);
		
		if( q.z == 0 ) {
			// no depth info, use 2d point
			float dx = u - q.x, dy = v - q.y;
			hx[i] = (dx * dx + dy * dy);
		}
		else {
			// use 3d point
			float dx = qx - px, dy = qy - py, dz = qz - pz;
			hx[i] = (dx * dx + dy * dy + dz * dz);// * wpt;
		}
		*/
		float dx = q.x - u, dy = q.y - v, dz = q.z==0?0:(q.z - d);
		hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
	}
}

void evalJacobian_2D(float *p, float *J, int m, int n, void *adata) {
	// J is a n-by-m matrix
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	// for Jacobian of projection/viewport transformation
	const float f_x = 525, f_y = 525;
	float Jf[6] = {0};

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];

		// point p
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);

		// point q
		const PhGUtils::Point3f& q = targets[i].first;

		// R * p
		float rpx = px, rpy = py, rpz = pz;
		PhGUtils::rotatePoint( rpx, rpy, rpz, R );

		// s * R * p + t
		float pkx = s * rpx + tx, pky = s * rpy + ty, pkz = s * rpz + tz;

		// Jf
		float inv_z = 1.0 / pkz;
		float inv_z2 = inv_z * inv_z;
		Jf[0] = f_x * inv_z; Jf[2] = -f_x * pkx * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;

		// project p to color image plane
		float pu, pv, pd;
		PhGUtils::worldToColor(pkx, pky, pkz, pu, pv, pd);

		// residue
		float rkx = pu - q.x, rky = pv - q.y;

		// J_? * p_k
		float jpx, jpy, jpz;
		// J_f * J_? * p_k
		float jjpx, jjpy;
		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		jjpx = Jf[0] * jpx + J[2] * jpz;
		jjpy = Jf[4] * jpy + J[5] * jpz;
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = wpt * 2.0 * s * (jjpx * rkx + jjpy * rky);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		jjpx = Jf[0] * jpx + J[2] * jpz;
		jjpy = Jf[4] * jpy + J[5] * jpz;
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = wpt * 2.0 * s * (jjpx * rkx + jjpy * rky);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		jjpx = Jf[0] * jpx + J[2] * jpz;
		jjpy = Jf[4] * jpy + J[5] * jpz;
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = wpt * 2.0 * s * (jjpx * rkx + jjpy * rky);

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = wpt * 2.0 * (Jf[0] * rkx);

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = wpt * 2.0 * (Jf[4] * rky);

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = wpt * 2.0 * (Jf[2] * rkx + Jf[5] * rky);

		// \frac{\partial r_i}{\partial s}
		jjpx = Jf[0] * rpx + J[2] * rpz;
		jjpy = Jf[4] * rpy + J[5] * rpz;
		J[jidx++] = wpt * 2.0 * (jjpx * rkx + jjpy * rky);
	}
}

void evalCost(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++) {
		//PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		float wpt = w_landmarks[vidx];
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		const PhGUtils::Point3f& q = targets[i].first;

		//PhGUtils::Point3f pp = R * p + T;
		PhGUtils::transformPoint( px, py, pz, R, T );

		//hx[i] = p.distanceTo(q) * w_landmarks[vidx];
		float dx = px - q.x, dy = py - q.y, dz = pz - q.z;
		hx[i] = (dx * dx + dy * dy + dz * dz) * wpt;
	}
}

void evalJacobian(float *p, float *J, int m, int n, void *adata) {
	// J is a n-by-m matrix
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];

		// point p
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		// point q
		const PhGUtils::Point3f& q = targets[i].first;

		// R * p
		float rpx = px, rpy = py, rpz = pz;
		PhGUtils::rotatePoint( rpx, rpy, rpz, R );

		// s * R * p + t - q
		float rkx = s * rpx + tx - q.x, rky = s * rpy + ty - q.y, rkz = s * rpz + tz - q.z;

		float jpx, jpy, jpz;
		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = wpt * 2.0 * rkx;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = wpt * 2.0 * rky;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = wpt * 2.0 * rkz;

		// \frac{\partial r_i}{\partial s}
		J[jidx++] = wpt * 2.0 * (rpx * rkx + rpy * rky + rpz * rkz);
	}
}

#if USE_CMINPACK
// function to evaluate residue when using cminpack
int evalCost_minpack(void *adata, int m, int n, const __cminpack_real__ *p, __cminpack_real__ *hx,
					 int iflag) 
{
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		const PhGUtils::Point3f& q = targets[i].first;

		//PhGUtils::Point3f pp = R * p + T;
		PhGUtils::transformPoint( p.x, p.y, p.z, R, T );

		hx[i] = p.distanceTo(q) * w_landmarks[vidx];
	}

	return 0;
}
#endif

bool MultilinearReconstructor::fitRigidTransformationAndScale_2D() {
	int npts = targets.size();
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	// use levmar
	int iters = slevmar_dif(evalCost_2D, RTparams, &(pws.meas[0]), 7, npts, 128, NULL, NULL, NULL, NULL, this);
	//int iters = slevmar_der(evalCost_2D, evalJacobian_2D, RTparams, &(pws.meas[0]), 7, npts, 128, opts, NULL, NULL, NULL, this);
	//PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");

	// set up the matrix and translation vector
	Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]) * RTparams[6];
	float diff = 0;
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			diff += fabs(R(i, j) - Rmat(i, j));
			R(i, j) = Rmat(i, j);			
		}
	}

	Tvec.x = RTparams[3], Tvec.y = RTparams[4], Tvec.z = RTparams[5];
	diff += fabs(Tvec.x - T(0)) + fabs(Tvec.y - T(1)) + fabs(Tvec.z - T(2));
	T(0) = RTparams[3], T(1) = RTparams[4], T(2) = RTparams[5];

	return diff / 7 < cc;
}

bool MultilinearReconstructor::fitRigidTransformationAndScale() {
	int npts = targets.size();
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	// use levmar
	//int iters = slevmar_dif(evalCost, RTparams, &(pws.meas[0]), 7, npts, 128, NULL, NULL, NULL, NULL, this);
	int iters = slevmar_der(evalCost, evalJacobian, RTparams, 
		&(pws.meas[0]), 7, npts, 128, opts, NULL, NULL, NULL, this);
	//PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");

	// use minpack
	//int iters = __cminpack_func__(lmdif1)(evalCost_minpack, this, npts, 7, RTparams, &(meas[0]), 1e-6, &(workspace[0]), &(w2[0]), w2.size());
	//PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");

	// set up the matrix and translation vector
	Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]) * RTparams[6];
	float diff = 0;
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			diff += fabs(R(i, j) - Rmat(i, j));
			R(i, j) = Rmat(i, j);			
		}
	}

	Tvec.x = RTparams[3], Tvec.y = RTparams[4], Tvec.z = RTparams[5];
	diff += fabs(Tvec.x - T(0)) + fabs(Tvec.y - T(1)) + fabs(Tvec.z - T(2));
	T(0) = RTparams[3], T(1) = RTparams[4], T(2) = RTparams[5];

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
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm1c(j, vidx) * p[j];
			y += tm1c(j, vidx+1) * p[j];
			z += tm1c(j, vidx+2) * p[j];
		}
		PhGUtils::Point3f p(x, y, z);
		const PhGUtils::Point3f& q = targets[i].first;

		PhGUtils::Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

bool MultilinearReconstructor::fitIdentityWeights_withPrior_2D()
{
	// to use this method, the tensor tm1c must first be updated using the rotation matrix and translation vector
	int nparams = core.dim(0);	

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	for(int i=0;i<tm1cRT.dim(0);i++) {
		for(int j=0;j<tm1cRT.dim(1);j++) {
			Aid(j, i) = tm1cRT(i, j);
		}
	}

	for(int j=0;j<Aid.cols();j++) {
		for(int i=0, ridx=tm1c.dim(1);i<nparams;i++, ridx++) {
			Aid(ridx, j) = sigma_wid_weighted(i, j);
		}
	}

	//PhGUtils::Matrix3x3f invRmat = Rmat.inv();

	int npts = targets.size();
	// assemble the right hand side, fill in the upper part as usual
	for(int i=0, vidx=0;i<npts;vidx+=3, i++) {
		brhs(vidx) = (q(vidx) - Tvec.x) * w_landmarks[vidx];
		brhs(vidx+1) = (q(vidx+1) - Tvec.y) * w_landmarks[vidx+1];
		brhs(vidx+2) = (q(vidx+2) - Tvec.z) * w_landmarks[vidx+2];
	}
	// fill in the lower part with the mean vector of identity weights
	int ndim_id = mu_wid.size();
	for(int i=0, idx=q.length();i<ndim_id;i++,idx++) {
		brhs(idx) = mu_wid_weighted(i);
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

	return diff / nparams < cc;
}

bool MultilinearReconstructor::fitIdentityWeights_withPrior()
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

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	for(int i=0;i<tm1cRT.dim(0);i++) {
		for(int j=0;j<tm1cRT.dim(1);j++) {
			Aid(j, i) = tm1cRT(i, j);
		}
	}

	for(int j=0;j<Aid.cols();j++) {
		for(int i=0, ridx=tm1c.dim(1);i<nparams;i++, ridx++) {
			Aid(ridx, j) = sigma_wid_weighted(i, j);
		}
	}

	//PhGUtils::Matrix3x3f invRmat = Rmat.inv();

	int npts = targets.size();
	// assemble the right hand side, fill in the upper part as usual
	for(int i=0, vidx=0;i<npts;vidx+=3, i++) {
		brhs(vidx) = (q(vidx) - Tvec.x) * w_landmarks[vidx];
		brhs(vidx+1) = (q(vidx+1) - Tvec.y) * w_landmarks[vidx+1];
		brhs(vidx+2) = (q(vidx+2) - Tvec.z) * w_landmarks[vidx+2];
	}
	// fill in the lower part with the mean vector of identity weights
	int ndim_id = mu_wid.size();
	for(int i=0, idx=q.length();i<ndim_id;i++,idx++) {
		brhs(idx) = mu_wid_weighted(i);
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
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm0c(j, vidx) * p[j];
			y += tm0c(j, vidx+1) * p[j];
			z += tm0c(j, vidx+2) * p[j];
		}
		PhGUtils::Point3f p(x, y, z);
		const PhGUtils::Point3f& q = targets[i].first;

		PhGUtils::Point3f pp = R * p + T;

		hx[i] = pp.distanceTo(q);
	}
}

bool MultilinearReconstructor::fitExpressionWeights_withPrior_2D()
{
	int nparams = core.dim(1);
	int npts = q.length() / 3;

	// fill in the upper part of the matrix, the lower part is already filled
	for(int i=0;i<tm0cRT.dim(0);i++) {
		for(int j=0;j<npts;j++) {
			Aexp(j, i) = tm0cRT(i, j);// * w_landmarks[j];
		}
	}

	// fill in the lower part
	for(int j=0;j<Aexp.cols();j++) {
		for(int i=0, ridx=tm0c.dim(1);i<nparams;i++, ridx++) {
			Aexp(ridx, j) = sigma_wexp_weighted(i, j);
		}
	}

	// fill in the upper part of the right hand side
	for(int i=0;i<q.length();i++) {
		brhs(i) = q(i) * w_landmarks[i];
	}

	// fill in the lower part with the mean vector of expression weights
	int ndim_exp = mu_wexp.size();
	for(int i=0, idx=q.length();i<ndim_exp;i++,idx++) {
		brhs(idx) = mu_wexp_weighted(i);
	}

#if USE_MKL_LS
	int rtn = leastsquare<float>(Aexp, brhs);
	//debug("rtn", rtn);

	//b.print("b");
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wexp(i) - brhs(i));
		Wexp(i) = brhs(i);
	}
#else
	int rtn = leastsquare_normalmat(Aexp, brhs, AexptAexp, Aexptb);
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		//cout << brhs(i) << ", " << Aexptb(i) << "\tdiff #" << i << " = " << fabs(brhs(i) - Aexptb(i)) << endl;
		diff += fabs(Wexp(i) - Aexptb(i));
		Wexp(i) = Aexptb(i);
		//cout << params[i] << ' ';
	}
	//PhGUtils::message("done");
	//::system("pause");
#endif

	return diff / nparams < cc;
}

bool MultilinearReconstructor::fitExpressionWeights_withPrior()
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
	int npts = tm0cRT.dim(1) / 3;

	// fill in the upper part of the matrix, the lower part is already filled
	for(int i=0;i<tm0cRT.dim(0);i++) {
		for(int j=0;j<tm0cRT.dim(1);j++) {
			Aexp(j, i) = tm0cRT(i, j);
		}
	}

	// fill in the lower part
	for(int j=0;j<Aexp.cols();j++) {
		for(int i=0, ridx=tm0c.dim(1);i<nparams;i++, ridx++) {
			Aexp(ridx, j) = sigma_wexp_weighted(i, j);
		}
	}

	// fill in the upper part of the right hand side
	for(int i=0, vidx=0;i<npts;i++,vidx+=3) {
		brhs(vidx) = (q(vidx) - Tvec.x) * w_landmarks[vidx];
		brhs(vidx+1) = (q(vidx+1) - Tvec.y) * w_landmarks[vidx];
		brhs(vidx+2) = (q(vidx+2) - Tvec.z) * w_landmarks[vidx];
	}

	// fill in the lower part with the mean vector of expression weights
	int ndim_exp = mu_wexp.size();
	for(int i=0, idx=q.length();i<ndim_exp;i++,idx++) {
		brhs(idx) = mu_wexp_weighted(i);
	}

#if USE_MKL_LS
	int rtn = leastsquare<float>(Aexp, brhs);
	//debug("rtn", rtn);

	//b.print("b");
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wexp(i) - brhs(i));
		Wexp(i) = brhs(i);
	}
#else
	int rtn = leastsquare_normalmat(Aexp, brhs, AexptAexp, Aexptb);
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		//cout << brhs(i) << ", " << Aexptb(i) << "\tdiff #" << i << " = " << fabs(brhs(i) - Aexptb(i)) << endl;
		diff += fabs(Wexp(i) - Aexptb(i));
		Wexp(i) = Aexptb(i);
		//cout << params[i] << ' ';
	}
	//PhGUtils::message("done");
	//::system("pause");
#endif

	// normalize Wexp

	//cout << endl;
	//cout << endl;
#endif

	return diff / nparams < cc;
}

// transforms the template mesh into a target mesh with rotation and translation
void MultilinearReconstructor::transformMesh()
{
	int nverts = tplt.length()/3;
	arma::fmat pt(3, nverts);
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		pt(0, i) = tplt(idx);
		pt(1, i) = tplt(idx+1);
		pt(2, i) = tplt(idx+2);
	}

	// batch rotation processing
	arma::fmat pt_trans =  R * pt;
	for(int i=0, idx=0;i<nverts;i++,idx+=3) {
		tmesh(idx) = pt_trans(0, i) + T(0);
		tmesh(idx+1) = pt_trans(1, i) + T(1);
		tmesh(idx+2) = pt_trans(2, i) + T(2);
	}
}

void MultilinearReconstructor::bindTarget( const vector<pair<PhGUtils::Point3f, int>>& pts, TargetType ttp )
{
	bool updateTC = false;
		// check if the computation tensors need update
	if( targets.size() != pts.size() )
		updateTC = true;
	else {
		// check if the vertex indices are the same
		for(int i=0;i<pts.size();i++) {
			if( pts[i].second != targets[i].second ) {
				updateTC = true;
				break;
			}
		}
	}

	targets = pts;
	int npts = targets.size();

	const float DEPTH_THRES = 1e-6;
	int validCount = 0;
	meanZ = 0;
	// initialize q
	q.resize(npts*3);	
	for(int i=0, idx=0;i<npts;i++, idx+=3) {
		int vidx = targets[i].second;
		const PhGUtils::Point3f& p = targets[i].first;	
		q(idx) = p.x;
		q(idx+1) = p.y;
		q(idx+2) = p.z;

		int isValid = (fabs(p.z) > DEPTH_THRES)?1:0;

		meanX += p.x * isValid;
		meanY += p.y * isValid;
		meanZ += p.z * isValid;

		if( ttp == TargetType_2D ) {
			w_landmarks[idx] = w_landmarks[idx+1] = w_landmarks[idx+2] = (i<64 || i>74)?1:w_boundary;
		}
		else {
			// set the landmark weights
			w_landmarks[idx] = w_landmarks[idx+1] = w_landmarks[idx+2] = (i<64 || i>74)?isValid:isValid*w_boundary;
		}

		validCount += isValid;
	}

	meanX /= validCount;
	meanY /= validCount;
	meanZ /= validCount;

	if( updateTC ) {
		pws.meas.resize(npts);
		updateComputationTensor();
		updateMatrices();

		// initialize rotation and translation

		switch( ttp ) {
		case TargetType_2D: 
			{
				float mx, my, mz;
				float lx, ly, lz, rx, ry, rz;
				PhGUtils::colorToWorld(targets[64].first.x, targets[64].first.y, targets[64].first.z, lx, ly, lz);
				PhGUtils::colorToWorld(targets[74].first.x, targets[74].first.y, targets[74].first.z, rx, ry, rz);
				PhGUtils::colorToWorld(meanX, meanY, meanZ, mx, my, mz);

				RTparams[3] = mx;
				RTparams[4] = my;
				RTparams[5] = mz;
				RTparams[6] = fabs(lx - rx);
				break;	
			}
		case TargetType_3D:
			{
				RTparams[3] = meanX;
				RTparams[4] = meanY;
				RTparams[5] = meanZ;
				RTparams[6] = fabs(targets[64].first.x - targets[74].first.x);
				break;
			}
		default:
			break;
		}

	}

	//PhGUtils::debug("valid landmarks", validCount);
}

void MultilinearReconstructor::updateMatrices() {
	if( usePrior ) {
		int ndim_id = mu_wid.size();
		int ndim_exp = mu_wexp.size();

		// extend the matrix with the prior term
		Aid = PhGUtils::DenseMatrix<float>::zeros(tm1c.dim(1) + ndim_id, tm1c.dim(0));
		Aexp = PhGUtils::DenseMatrix<float>::zeros(tm0c.dim(1) + ndim_exp, tm0c.dim(0));

		AidtAid = PhGUtils::DenseMatrix<float>::zeros(tm1c.dim(0), tm1c.dim(0));
		AexptAexp = PhGUtils::DenseMatrix<float>::zeros(tm0c.dim(0), tm0c.dim(0));

		// take the larger size for the right hand side vector
		brhs.resize(targets.size()*3 + max(ndim_id, ndim_exp));
		Aidtb.resize(tm1c.dim(0));
		Aexptb.resize(tm0c.dim(0));
	}
	else {
		Aid = PhGUtils::DenseMatrix<float>::zeros(tm1c.dim(1), tm1c.dim(0));
		Aexp = PhGUtils::DenseMatrix<float>::zeros(tm0c.dim(1), tm0c.dim(0));

		AidtAid = PhGUtils::DenseMatrix<float>::zeros(tm1c.dim(0), tm1c.dim(0));
		AexptAexp = PhGUtils::DenseMatrix<float>::zeros(tm0c.dim(0), tm0c.dim(0));

		brhs.resize(targets.size()*3);
		Aidtb.resize(tm1c.dim(0));
		Aexptb.resize(tm0c.dim(0));
	}
}

void MultilinearReconstructor::updateComputationTensor()
{
	updateCoreC();
	tm0c = corec.modeProduct(Wid, 0);
	tm0cRT = tm0c;
	tm1c = corec.modeProduct(Wexp, 1);
	tm1cRT = tm1c;
	tmc = tm1c.modeProduct(Wid, 0);
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
	// don't use the assignment operator, it actually moves the data
	//tm0cRT = tm0c;
	for(int i=0;i<tm0c.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {

			tm0cRT(i, vidx) = tm0c(i, vidx);
			tm0cRT(i, vidx+1) = tm0c(i, vidx+1);
			tm0cRT(i, vidx+2) = tm0c(i, vidx+2);
			// store the rotated tensor in tm0cRT
			PhGUtils::rotatePoint( tm0cRT(i, vidx), tm0cRT(i, vidx+1), tm0cRT(i, vidx+2), Rmat );

			tm0cRT(i, vidx) *= w_landmarks[vidx];
			tm0cRT(i, vidx+1) *= w_landmarks[vidx+1];
			tm0cRT(i, vidx+2) *= w_landmarks[vidx+2];
		}
	}
}

// transform TM1C with global rigid transformation
void MultilinearReconstructor::transformTM1C() {
	int npts = tm1c.dim(1) / 3;
	// don't use the assignment operator, it actually moves the data
	//tm1cRT = tm1c;
	for(int i=0;i<tm1c.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {

			tm1cRT(i, vidx) = tm1c(i, vidx);
			tm1cRT(i, vidx+1) = tm1c(i, vidx+1);
			tm1cRT(i, vidx+2) = tm1c(i, vidx+2);

			// rotation only!!!
			PhGUtils::rotatePoint( tm1cRT(i, vidx), tm1cRT(i, vidx+1), tm1cRT(i, vidx+2), Rmat );
		
			// multiply weights
			tm1cRT(i, vidx) *= w_landmarks[vidx];
			tm1cRT(i, vidx+1) *= w_landmarks[vidx+1];
			tm1cRT(i, vidx+2) *= w_landmarks[vidx+2];
		}
	}
}

void MultilinearReconstructor::updateTMC() {
	tm1c.modeProduct(Wid, 0, tmc);
}

void MultilinearReconstructor::updateTMCwithTM0C() {
	//PhGUtils::message("updating tmc");
	tm0c.modeProduct(Wexp, 0, tmc);
}

float MultilinearReconstructor::computeError()
{
	int npts = targets.size();
	float E = 0;
	for(int i=0;i<npts;i++) {
		int vidx = i * 3;
		// should change this to a fast version
		PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));

		/*p = Rmat * p + Tvec;*/		
		PhGUtils::transformPoint(p.x, p.y, p.z, Rmat, Tvec);

		E += p.squaredDistanceTo(targets[i].first) * w_landmarks[vidx];
	}

	E /= npts;
	return E;
}

float MultilinearReconstructor::computeError_2D()
{
	int npts = targets.size();
	float E = 0;
	for(int i=0;i<npts;i++) {
		int vidx = i * 3;
		// should change this to a fast version
		PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));

		/*p = Rmat * p + Tvec;*/		
		PhGUtils::transformPoint(p.x, p.y, p.z, Rmat, Tvec);

		// project the point
		float u, v, d;
		PhGUtils::worldToColor(p.x, p.y, p.z, u, v, d);

		float dx, dy;
		dx = u - targets[i].first.x;
		dy = v - targets[i].first.y;
		E += (dx * dx + dy * dy) * w_landmarks[vidx];
	}

	E /= npts;
	return E;
}

#include "MultilinearReconstructor.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Utils/fileutils.h"
#include "Kinect/KinectUtils.h"
#include "Math/denseblas.h"
#include "Math/Optimization.hpp"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#include "Geometry/geometryutils.hpp"

#define USELEVMAR4WEIGHTS 0
#define USE_MKL_LS 0		// use mkl least square solver
#define OUTPUT_STATS 0
#define FBO_DEBUG 0

MultilinearReconstructor::MultilinearReconstructor(void)
{	
	w_prior_id = 1e-3;//1e-3;
	// for 25 expression dimensions
	//w_prior_exp = 1e-4;
	// for 47 expression dimensions
	w_prior_exp = 5e-4;

	// for outer contour and chin region, use only 2D feature point info
	w_boundary = 1e-8;
	w_chin = 1e-6;
	w_outer = 1e2;
	w_fp = 2.5;

	w_prior_exp_2D = 5e3;
	w_prior_id_2D = 5e3;
	w_boundary_2D = 1.0;

	w_history = 1e-3;

	w_ICP = 1.5;

	meanX = meanY = meanZ = 0;

	mkl_set_num_threads(8);

	loadCoreTensor();
	loadPrior();
	initializeWeights();
	createTemplateItem();
	//updateComputationTensor();
	init();

	w_landmarks.resize(512);

	// off-screen rendering related
	depthMap.resize(640*480);
	indexMap.resize(640*480*4);
	faceMask.resize(640*480);
	mProj = PhGUtils::KinectColorProjection.transposed();
	mMv = PhGUtils::Matrix4x4f::identity();


	dummyWgt = shared_ptr<QGLWidget>(new QGLWidget());
	dummyWgt->hide();
	dummyWgt->makeCurrent();
	fbo = shared_ptr<QGLFramebufferObject>(new QGLFramebufferObject(640, 480, QGLFramebufferObject::Depth));
	dummyWgt->doneCurrent();

	// for ICP
	targetLocations.resize(640*480);


	// convergence
	cc = 1e-4;
	errorThreshold = 2.5e-3;
	errorDiffThreshold = errorThreshold * 0.005;
	usePrior = true;

	frameCounter = 0;

	useHistory = true;
	historyWeights[0] = 0.02;
	historyWeights[1] = 0.04;
	historyWeights[2] = 0.08;
	historyWeights[3] = 0.16;
	historyWeights[4] = 0.32;
	historyWeights[5] = 0.64;
	historyWeights[6] = 1.28;
	historyWeights[7] = 2.56;
	historyWeights[8] = 5.12;
	historyWeights[9] = 10.24;
}

MultilinearReconstructor::~MultilinearReconstructor(void)
{
}


void MultilinearReconstructor::reset()
{
	// @TODO	reset the reconstructor
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
	mu_wid_orig = mu_wid;
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
	mu_wexp_orig = mu_wexp;
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
	tm0RT = tm0;
	tm1 = core.modeProduct(Wexp, 1);
	tm1RT = tm1;
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

void MultilinearReconstructor::fitICP(FittingOption ops /*= FIT_ALL*/)
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
			fitScale = true;
			fitPose = true;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitScale = false;
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
		fitICP_withPrior();

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

		fitICP_withPrior();

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

// create a convex hull as the face mask
void MultilinearReconstructor::createFaceMask()
{
	vector<PhGUtils::Point2f> pts;
	for(int i=0;i<targets_2d.size();i++) pts.push_back(PhGUtils::Point2f(targets_2d[i].first.x, targets_2d[i].first.y));
	vector<PhGUtils::Point2f> hull = PhGUtils::convexHull(pts);
	
	//PhGUtils::printVector(hull);

	// for every pixel in the RGBD image, test if it is inside the convex hull
	for(int i=0, idx=0;i<480;i++) {
		for(int j=0;j<640;j++, idx++) {
			faceMask[idx] = PhGUtils::isInside(PhGUtils::Point2f(j, i), hull)?1:0;
		}
	}

	/*
	PhGUtils::write2file("hull.txt", [&](ofstream& fout){
		PhGUtils::printVector(hull, fout);
	});
	PhGUtils::write2file("mask.txt", [&](ofstream& fout){
		PhGUtils::dump2DArray(&(faceMask[0]), 480, 640, fout);
	});
	*/
}

void MultilinearReconstructor::collectICPConstraints()
{
	icpc.clear();
	// the depth map and the face index map are flipped vertically
	for(int v=0, vv=479, idx=0;v<480;v++, vv--) {
		for(int u=0;u<640;u++, idx++) {
			int didx = vv * 640 + u;					// pixel index for synthesized image
			// check if the synthesized depth is valid
			if( depthMap[didx] < 1.0 ) {
				const PhGUtils::Point3f& q = targetLocations[idx];
				auto inside = faceMask[idx];
				// check if the input location is valid
				if( q.z == 0 || inside == 0 ) continue;

				// take a small window
				const int wSize = 3;
				set<int> checkedFaces;

				float closestDist = numeric_limits<float>::max();
				int closestVerts[3];
				PhGUtils::Point3f closestHit;

				for(int r=v-wSize;r<=v+wSize;r++) {
					int rr = 479 - r;
					for(int c=u-wSize;c<=u+wSize;c++) {
						int pidx = rr * 640 + c;
						int poffset = pidx << 2;		// pixel index for synthesized image

						// get face index and depth value
						int fidx;
						PhGUtils::decodeIndex<float>(indexMap[poffset]/255.0f, indexMap[poffset+1]/255.0f, indexMap[poffset+2]/255.0f, fidx);
						float depthVal = depthMap[pidx];

						if( depthVal < 1.0 ) {
							if( std::find(checkedFaces.begin(), checkedFaces.end(), fidx) == checkedFaces.end() ) {
								// not checked yet
								checkedFaces.insert(fidx);

								const PhGUtils::QuadMesh::face_t& f = baseMesh.face(fidx);
								PhGUtils::Point3f hit1, hit2;
								// find the closest point
								float dist1 = PhGUtils::pointToTriangleDistance(
									q,
									baseMesh.vertex(f.x), baseMesh.vertex(f.y), baseMesh.vertex(f.z),
									hit1
									);
								float dist2 = PhGUtils::pointToTriangleDistance(
									q,
									baseMesh.vertex(f.y), baseMesh.vertex(f.z), baseMesh.vertex(f.w),
									hit2
									);

								// take the smaller one
								if( dist1 < dist2 && dist1 < closestDist) {
									closestDist = dist1;
									closestVerts[0] = f.x, closestVerts[1] = f.y, closestVerts[2] = f.z;
									closestHit = hit1;
								}
								else if( dist2 < closestDist ) {
									closestDist = dist2;
									closestVerts[0] = f.y, closestVerts[1] = f.z, closestVerts[2] = f.w;
									closestHit = hit2;
								}
							}
							else {
								// already checked, do nothing
							}
						}						
					}
				}

				const float DIST_THRES = 0.025;				
				// close enough to be a constraint
				if( closestDist < DIST_THRES ) {
					ICPConstraint cc;
					cc.q = q;
					cc.v[0] = closestVerts[0], cc.v[1] = closestVerts[1], cc.v[2] = closestVerts[2];
					PhGUtils::computeBarycentricCoordinates(
						closestHit,
						baseMesh.vertex(closestVerts[0]), baseMesh.vertex(closestVerts[1]), baseMesh.vertex(closestVerts[2]), 
						cc.bcoords);

					icpc.push_back(cc);
				}
			}
		}
	}

	cout << "ICP constraints: " << icpc.size() << endl;

	/*
	// output ICP constraints to file
	ofstream fout("icpc.txt");
	for(int i=0;i<icpc.size();i++) {
		PhGUtils::Point3f p(0, 0, 0);
		p = p + icpc[i].bcoords[0] * baseMesh.vertex(icpc[i].v[0]);
		p = p + icpc[i].bcoords[1] * baseMesh.vertex(icpc[i].v[1]);
		p = p + icpc[i].bcoords[2] * baseMesh.vertex(icpc[i].v[2]);
		fout << icpc[i].q << " " << p << endl;
	}
	fout.close();

	::system("pause");
	*/
}

void MultilinearReconstructor::collectICPConstraints_topo()
{
	icpc.clear();
	// the depth map and the face index map are flipped vertically
	for(int v=0, vv=479, idx=0;v<480;v++, vv--) {
		for(int u=0;u<640;u++, idx++) {
			int didx = vv * 640 + u;					// pixel index for synthesized image
			// check if the synthesized depth is valid
			if( depthMap[didx] < 1.0 ) {
				const PhGUtils::Point3f& q = targetLocations[idx];
				unsigned char inside = faceMask[idx];

				// check if the input location is valid
				if( q.z == 0 || inside == 0 ) continue;

				float closestDist = numeric_limits<float>::max();
				int closestVerts[3];
				PhGUtils::Point3f closestHit;

				set<int> checkedFaces;
				vector<int> facesToCheck;

				const int wSize = 3;
				for(int r=v-wSize;r<=v+wSize;r++) {
					int rr = 479 - r;
					for(int c=u-wSize;c<=u+wSize;c++) {
						int pidx = rr * 640 + c;
						int poffset = pidx << 2;		// pixel index for synthesized image

						int fidx;
						PhGUtils::decodeIndex<float>(indexMap[poffset]/255.0f, indexMap[poffset+1]/255.0f, indexMap[poffset+2]/255.0f, fidx);
						float depthVal = depthMap[pidx];

						if( depthVal < 1.0 ) {
							queue<pair<int, int>> candidates;	// face/level pair
							candidates.push(make_pair(fidx, 0));
							checkedFaces.insert(fidx);

							int maxLev = 3;
							// find faces within certain levels
							while( !candidates.empty() ) {
								int curFace = candidates.front().first;
								int curLev = candidates.front().second;
								candidates.pop();
								
								facesToCheck.push_back(curFace);

								if( curLev < maxLev ) {
									// push not checked faces
									// for all vertices of current face, push non-checked incident faces
									const PhGUtils::QuadMesh::face_t& parentFace = baseMesh.face(curFace);
									int v[4] = {parentFace.x, parentFace.y, parentFace.z, parentFace.w};
									for(int vi=0;vi<4;vi++) {
										const set<int>& incidentFaces = baseMesh.incidentFaces(v[vi]);
										for(auto sit=incidentFaces.begin();sit!=incidentFaces.end();++sit) {
											if( checkedFaces.find((*sit)) == checkedFaces.end() ) {
												candidates.push(make_pair(*sit, curLev+1));

												checkedFaces.insert(*sit);
											}
										}
									}
								}
							}

							// check the faces found
							for(auto fit=facesToCheck.begin();fit!=facesToCheck.end();fit++) {

								const PhGUtils::QuadMesh::face_t& f = baseMesh.face(*fit);

								PhGUtils::Point3f hit1, hit2;
								// find the closest point
								float dist1 = PhGUtils::pointToTriangleDistance(
									q,
									baseMesh.vertex(f.x), baseMesh.vertex(f.y), baseMesh.vertex(f.z),
									hit1
									);
								float dist2 = PhGUtils::pointToTriangleDistance(
									q,
									baseMesh.vertex(f.y), baseMesh.vertex(f.z), baseMesh.vertex(f.w),
									hit2
									);

								// take the smaller one
								if( dist1 < dist2 && dist1 < closestDist) {
									closestDist = dist1;
									closestVerts[0] = f.x, closestVerts[1] = f.y, closestVerts[2] = f.z;
									closestHit = hit1;
								}
								else if( dist2 < closestDist ) {
									closestDist = dist2;
									closestVerts[0] = f.y, closestVerts[1] = f.z, closestVerts[2] = f.w;
									closestHit = hit2;
								}
							}
						}				
					}
				}

				const float DIST_THRES = 0.025;				
				// close enough to be a constraint
				if( closestDist < DIST_THRES ) {
					ICPConstraint cc;
					cc.q = q;
					cc.v[0] = closestVerts[0], cc.v[1] = closestVerts[1], cc.v[2] = closestVerts[2];
					PhGUtils::computeBarycentricCoordinates(
						closestHit,
						baseMesh.vertex(closestVerts[0]), baseMesh.vertex(closestVerts[1]), baseMesh.vertex(closestVerts[2]), 
						cc.bcoords);

					icpc.push_back(cc);
				}
			}
		}
	}
	/*
	cout << "ICP constraints: " << icpc.size() << endl;

	
	// output ICP constraints to file
	ofstream fout("icpc.txt");
	for(int i=0;i<icpc.size();i++) {
		PhGUtils::Point3f p(0, 0, 0);
		p = p + icpc[i].bcoords[0] * baseMesh.vertex(icpc[i].v[0]);
		p = p + icpc[i].bcoords[1] * baseMesh.vertex(icpc[i].v[1]);
		p = p + icpc[i].bcoords[2] * baseMesh.vertex(icpc[i].v[2]);
		fout << icpc[i].q << " " << p << endl;
	}
	fout.close();

	::system("pause");
	*/	
}

void MultilinearReconstructor::fitICP_withPrior() {
	PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

	cout << "initial guess ..." << endl;
	PhGUtils::printArray(RTparams, 7);
	// assemble initial guess and transform mesh
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
	
	timerTotal.tic();
	int iters = 0;
	float E0 = 0;
	bool converged = false;
	while( !converged && iters++ < 8 ) {
		converged = true;

		// update mesh and render the mesh
		transformMesh();
		updateMesh();
		renderMesh();

		// collect ICP constraints
		createFaceMask();
		collectICPConstraints();
		//collectICPConstraints_topo();


		if( fitPose ) {
			timerRT.tic();
			converged &= fitRigidTransformationAndScale_ICP();		
			timerRT.toc();
		}

		if( fitIdentity ) {			
			// apply the new global rotation to tm1c
			// because tm1c is required in fitting identity weights
			timerOther.tic();
			transformTM1();
			timerOther.toc();

			timerID.tic();
			converged &= fitIdentityWeights_withPrior_ICP();
			timerID.toc();
		}

		if( fitIdentity && (fitExpression || fitPose) ) {
			timerOther.tic();
			// update tm0 with the new identity weights
			// now the tensor is not updated with global rigid transformation
			tm0 = core.modeProduct(Wid, 0);
			timerOther.toc();
		}

		if( fitExpression ) {
			timerOther.tic();
			// apply the global rotation to tm0c
			// because tm0 is required in fitting expression weights
			transformTM0();
			timerOther.toc();

			timerExp.tic();
			converged &= fitExpressionWeights_withPrior_ICP();	
			timerExp.toc();
		}

		// this is not exactly logically correct
		// but this works for the case of fitting both pose and expression
		if( fitExpression && fitIdentity ) {//(fitIdentity || fitPose) ) {
			timerOther.tic();
			// update tm1 with the new expression weights
			// now the tensor is not updated with global rigid transformation
			tm1 = core.modeProduct(Wexp, 1);
			timerOther.toc();
		}

		//::system("pause");


		// compute tmc from the new tm1c or new tm0c
		if( fitIdentity ) {
			timerOther.tic();
			updateTM();
			timerOther.toc();
		}
		else if( fitExpression ) {
			timerOther.tic();
			updateTMwithTM0();
			//updateTMC();
			timerOther.toc();
		}		

		timerOther.tic();
		// uncomment to show the transformation process
		//transformMesh();
		//Rmat.print("R");
		//Tvec.print("T");
		E = computeError();
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
		vector<float> mRT = computeWeightedMeanPose();
		for(int i=0;i<7;i++) meanRT[i] = mRT[i];
	}

	timerTransform.tic();
	//PhGUtils::debug("R", Rmat);
	//PhGUtils::debug("T", Tvec);
	transformMesh();
	timerTransform.toc();
	//emit oneiter();
	timerTotal.toc();

	renderMesh();

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

		E = computeError_2D();
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
			fitScale = true;
			fitPose = true;
			fitIdentity = true;
			fitExpression = false;
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitScale = false;
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
		E = computeError();
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
		vector<float> mRT = computeWeightedMeanPose();
		for(int i=0;i<7;i++) meanRT[i] = mRT[i];
	}

	timerTransform.tic();
	//PhGUtils::debug("R", Rmat);
	//PhGUtils::debug("T", Tvec);
	transformMesh();
	timerTransform.toc();
	//emit oneiter();
	timerTotal.toc();

	//renderMesh();

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

void evalCost_ICP(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto icpc = recon->icpc;
	int npts_ICP = icpc.size();
	auto fp = recon->targets;
	auto fp2d = recon->targets_2d;
	int npts_feature = fp.size();
	auto tplt = recon->tplt;

	// set up rotation matrix and translation vector
	auto w_history = recon->w_history;
	auto meanRT = recon->meanRT;
	
	auto w_ICP = recon->w_ICP;
	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;
	auto w_outer = recon->w_outer;
	auto w_fp = recon->w_fp;

	auto fitScale = recon->fitScale;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// ICP terms
	for(int i=0;i<npts_ICP;i++) {
		auto v = icpc[i].v;
		auto bcoords = icpc[i].bcoords;

		int vidx[3];
		vidx[0] = v[0]*3; vidx[1] = v[1]*3; vidx[2] = v[2]*3;

		PhGUtils::Point3f p;
		p.x += bcoords[0] * tplt(vidx[0]); p.y += bcoords[0] * tplt(vidx[0]+1); p.z += bcoords[0] * tplt(vidx[0]+2);
		p.x += bcoords[1] * tplt(vidx[1]); p.y += bcoords[1] * tplt(vidx[1]+1); p.z += bcoords[1] * tplt(vidx[1]+2);
		p.x += bcoords[2] * tplt(vidx[2]); p.y += bcoords[2] * tplt(vidx[2]+1); p.z += bcoords[2] * tplt(vidx[2]+2);

		const PhGUtils::Point3f& q = icpc[i].q;

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, R, T );

		float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
		hx[i] = (dx * dx + dy * dy + dz * dz) * w_ICP;
	}

	// facial feature term
	for(int i=0, hxidx = npts_ICP;i<npts_feature;i++, hxidx++) {
		int vidx = fp[i].second * 3;
		float wpt = w_landmarks[i*3];
		
		PhGUtils::Point3f p(tplt(vidx), tplt(vidx+1), tplt(vidx+2));

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, R, T );

		// for mouth region and outer contour, use only 2D info
		if( i < 42 || i > 74 ) {
			const PhGUtils::Point3f& q = fp[i].first;

			float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
			hx[hxidx] = (dx * dx + dy * dy + dz * dz) * wpt;
		}
		else {
			if( i > 63 ) {
				wpt *= w_outer;
			}
			else {
				wpt *= w_chin;
			}

			wpt *= w_fp;

			float u, v, d;
			PhGUtils::worldToColor(p.x, p.y, p.z, u, v, d);
			const PhGUtils::Point3f& q = fp2d[i].first;

			//cout << "#" << i <<"\t" << u << ", " << v << "\t" << q.x << ", " << q.y << endl;

			float du = u - q.x, dv = v - q.y;
			hx[hxidx] = (du * du + dv * dv) * wpt;
		}
	}
	
	// regularization terms
	int nterms = fitScale?7:6;
	for(int i=0, hxidx=npts_ICP+npts_feature;i<nterms;i++, hxidx++) {
		float diff = p[i] - meanRT[i];
		hx[hxidx] = diff * diff * w_history;
	}

	//::system("pause");
}

void evalJacobian_ICP(float *p, float *J, int m, int n, void *adata) {
	// J is a n-by-m matrix
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto icpc = recon->icpc;
	int npts_ICP = icpc.size();

	auto fp = recon->targets;
	auto fp2d = recon->targets_2d;
	int npts_feature = fp.size();

	auto tplt = recon->tplt;

	auto w_history = recon->w_history;
	auto meanRT = recon->meanRT;

	auto w_ICP = recon->w_ICP;
	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;
	auto w_outer = recon->w_outer;
	auto w_fp = recon->w_fp;

	auto fitScale = recon->fitScale;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	// ICP terms
	int jidx = 0;
	for(int i=0;i<npts_ICP;i++) {
		auto v = icpc[i].v;
		auto bcoords = icpc[i].bcoords;

		int vidx[3];
		vidx[0] = v[0]*3; vidx[1] = v[1]*3; vidx[2] = v[2]*3;

		// point p
		PhGUtils::Point3f p;
		p.x += bcoords[0] * tplt(vidx[0]); p.y += bcoords[0] * tplt(vidx[0]+1); p.z += bcoords[0] * tplt(vidx[0]+2);
		p.x += bcoords[1] * tplt(vidx[1]); p.y += bcoords[1] * tplt(vidx[1]+1); p.z += bcoords[1] * tplt(vidx[1]+2);
		p.x += bcoords[2] * tplt(vidx[2]); p.y += bcoords[2] * tplt(vidx[2]+1); p.z += bcoords[2] * tplt(vidx[2]+2);

		// point q
		const PhGUtils::Point3f& q = icpc[i].q;

		// R * p
		float rpx = p.x, rpy = p.y, rpz = p.z;
		PhGUtils::rotatePoint( rpx, rpy, rpz, R );

		// s * R * p + t - q
		float rkx = s * rpx + tx - q.x, rky = s * rpy + ty - q.y, rkz = s * rpz + tz - q.z;

		float jpx, jpy, jpz;
		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz) * w_ICP;

		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] =  2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz) * w_ICP;

		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz) * w_ICP;

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * rkx * w_ICP;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * rky * w_ICP;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * rkz * w_ICP;

		if( fitScale ) {
			// \frac{\partial r_i}{\partial s}
			J[jidx++] = 2.0 * (rpx * rkx + rpy * rky + rpz * rkz) * w_ICP;
		}
	}

	// facial feature terms
	for(int i=0;i<npts_feature;i++) {
		auto vidx = fp[i].second * 3;
		float wpt = w_landmarks[i*3];

		// point p
		PhGUtils::Point3f p(tplt(vidx), tplt(vidx+1), tplt(vidx+2));

		// for mouth region and outer contour, use only 2D points
		if( i < 42 || i > 74 ) {
			// point q
			const PhGUtils::Point3f& q = fp[i].first;

			// R * p
			float rpx = p.x, rpy = p.y, rpz = p.z;
			PhGUtils::rotatePoint( rpx, rpy, rpz, R );

			// s * R * p + t - q
			float rkx = s * rpx + tx - q.x, rky = s * rpy + ty - q.y, rkz = s * rpz + tz - q.z;

			float jpx, jpy, jpz;
			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
			// \frac{\partial r_i}{\partial \theta_x}
			J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
			// \frac{\partial r_i}{\partial \theta_y}
			J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
			// \frac{\partial r_i}{\partial \theta_z}
			J[jidx++] = wpt * 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

			// \frac{\partial r_i}{\partial \t_x}
			J[jidx++] = wpt * 2.0 * rkx;

			// \frac{\partial r_i}{\partial \t_y}
			J[jidx++] = wpt * 2.0 * rky;

			// \frac{\partial r_i}{\partial \t_z}
			J[jidx++] = wpt * 2.0 * rkz;

			if( fitScale ) {
				// \frac{\partial r_i}{\partial s}
				J[jidx++] = wpt * 2.0 * (rpx * rkx + rpy * rky + rpz * rkz);
			}
		}
		else {
			if( i > 63 ) {
				wpt *= w_outer;
			}
			else {
				wpt *= w_chin;
			}

			wpt *= w_fp;

			// point q
			const PhGUtils::Point3f& q = fp2d[i].first;

			// R * p
			float rpx = p.x, rpy = p.y, rpz = p.z;
			PhGUtils::rotatePoint( rpx, rpy, rpz, R );

			// s * R * p + t
			float pkx = s * rpx + tx, pky = s * rpy + ty, pkz = s * rpz + tz;

			// Jf
			float inv_z = 1.0 / pkz;
			float inv_z2 = inv_z * inv_z;
			/*
			Jf[0] = f_x * inv_z; Jf[2] = -f_x * pkx * inv_z2;
			Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
			*/

			const float f_x = 525.0, f_y = 525.0;
			float Jf[6] = {0};
			Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
			Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;

			// project p to color image plane
			float pu, pv, pd;
			PhGUtils::worldToColor(pkx, pky, pkz, pu, pv, pd);

			// residue
			float rkx = pu - q.x, rky = pv - q.y;

			// J_? * p_k
			float jpx, jpy, jpz;
			// J_f * J_? * p_k
			float jfjpx, jfjpy;

			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_x}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_y}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jpx = p.x, jpy = p.y, jpz = p.z;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_z}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_x}
			J[jidx++] = 2.0 * (Jf[0] * rkx) * wpt;

			// \frac{\partial r_i}{\partial \t_y}
			J[jidx++] = 2.0 * (Jf[4] * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_z}
			J[jidx++] = 2.0 * (Jf[2] * rkx + Jf[5] * rky) * wpt;

			if( fitScale ) {
				// \frac{\partial r_i}{\partial s}
				jfjpx = Jf[0] * rpx + Jf[2] * rpz;
				jfjpy = Jf[4] * rpy + Jf[5] * rpz;
				J[jidx++] = 2.0 * (jfjpx * rkx + jfjpy * rky) * wpt;
			}
		}
	}
		
	// regularization terms
	int nterms = fitScale?7:6;
	for(int i=0;i<nterms;i++) {
		float diff = p[i] - meanRT[i];
		if( diff == 0 ) diff = numeric_limits<float>::min();
		for(int j=0;j<nterms;j++) {
			if( j == i ) {
				J[jidx] = 2.0 * diff * w_history;
			}
			else J[jidx] = 0;
			jidx++;
		}
	}
}

bool MultilinearReconstructor::fitRigidTransformationAndScale_ICP() {
	int nparams = fitScale?7:6;
	int npts = icpc.size() + targets.size() + nparams;

	/*
	vector<float> meas(npts);
	// use levmar
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	//int iters = slevmar_dif(evalCost_ICP, RTparams, &(meas[0]), nparams, npts, 128, NULL, NULL, NULL, NULL, this);
	int iters = slevmar_der(evalCost_ICP, evalJacobian_ICP, RTparams, &(meas[0]), nparams, npts, 128, opts, NULL, NULL, NULL, this);
	*/
	
	
	// use Gauss-Newton
	float opts[3] = {0.1, 1e-3, 1e-4};		// why is the step size so small, must be something wrong with the Jacobian?
	int iters = PhGUtils::GaussNewton<float>(evalCost_ICP, evalJacobian_ICP, RTparams, NULL, NULL, nparams, npts, 128, opts, this);
	

	PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");
	
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

	
	cout << R << endl;
	cout << T << endl;
	

	return diff / 7 < cc;
}

void evalCost_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++) {
		//PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		float wpt = w_landmarks[vidx];

		// exclude the mouth region
		if( i>41 && i < 64 ) {
			wpt *= w_chin;
		}

		// use only 2D info for outer contour
		if( i >= 64 && i < 75 ) wpt = 0;

		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		const PhGUtils::Point3f& q = targets[i].first;

		// PhGUtils::Point3f pp = R * p + T;
		PhGUtils::transformPoint( px, py, pz, R, T );

		// Projection to image plane
		float u, v, d;
		PhGUtils::worldToColor( px, py, pz, u, v, d );

		//cout << i << "\t" << u << ", " << v << ", " << d << endl;

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
		float dx = q.x - u, dy = q.y - v, dz = (q.z==0?0:(q.z - d));
		//cout << i << "\t" << dx << ", " << dy << ", " << dz << endl;
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
	auto w_chin = recon->w_chin;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	//cout << Jx << endl;
	//cout << Jy << endl;
	//cout << Jz << endl;

	// for Jacobian of projection/viewport transformation
	const float f_x = 525, f_y = 525;
	float Jf[9] = {0};

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];

		// exclude the mouth region
		if( i>41 && i < 64 ) {
			wpt *= w_chin;
		}

		// use only 2D info for outer contour
		if( i >= 64 && i < 75 ) wpt = 0;

		// point p
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		//cout << px << ", " << py << ", " << pz << endl;

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
		/*
		Jf[0] = f_x * inv_z; Jf[2] = -f_x * pkx * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
		*/
		Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
		Jf[8] = -1000;

		// project p to color image plane
		float pu, pv, pd;
		PhGUtils::worldToColor(pkx, pky, pkz, pu, pv, pd);

		// residue
		float rkx = pu - q.x, rky = pv - q.y, rkz = (q.z==0?0:(pd - q.z));		

		// J_? * p_k
		float jpx, jpy, jpz;
		// J_f * J_? * p_k
		float jfjpx, jfjpy, jfjpz;

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		jfjpz = Jf[8] * jpz;
		/*
		cout << "jf\t";
		PhGUtils::printArray(Jf, 9);
		cout << "j\t" << jpx << ", " << jpy << ", "  << jpz << endl;
		cout << "jj\t" << jfjpx << ", " << jfjpy << ", "  << jfjpz << endl;
		*/

		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky + wpt * jfjpz * rkz);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		jfjpz = Jf[8] * jpz;
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky + wpt * jfjpz * rkz);

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		jfjpz = Jf[8] * jpz;
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky + wpt * jfjpz * rkz);

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * (Jf[0] * rkx);

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * (Jf[4] * rky);

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * (Jf[2] * rkx + Jf[5] * rky + wpt * Jf[8] * rkz);

		// \frac{\partial r_i}{\partial s}
		jfjpx = Jf[0] * rpx + Jf[2] * rpz;
		jfjpy = Jf[4] * rpy + Jf[5] * rpz;
		jfjpz = Jf[8] * rpz;
		J[jidx++] = 2.0 * (jfjpx * rkx + jfjpy * rky + wpt * jfjpz * rkz);
	}

	/*
	ofstream fout("jacobian.txt");

	for(int i=0, jidx=0;i<npts;i++) {
	for(int j=0;j<7;j++) {
	fout << J[jidx++] << '\t';
	}
	fout << endl;
	}
	fout.close();


	::system("pause");
	*/		
}

bool MultilinearReconstructor::fitRigidTransformationAndScale_2D() {
	int npts = targets.size();
	// use levmar

	/*
	vector<float> errs(npts);
	slevmar_chkjac(evalCost_2D, evalJacobian_2D, RTparams, 7, npts, this, &(errs[0]));
	PhGUtils::printVector(errs);
	::system("pause");
	*/

	vector<float> meas(npts);

	//int iters = slevmar_dif(evalCost_2D, RTparams, &(pws.meas[0]), 7, npts, 128, NULL, NULL, NULL, NULL, this);
	//float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	//int iters = slevmar_der(evalCost_2D, evalJacobian_2D, RTparams, &(meas[0]), 7, npts, 128, opts, NULL, NULL, NULL, this);

	float opts[3] = {0.1, 1e-3, 1e-4};
	int iters = PhGUtils::GaussNewton<float>(evalCost_2D, evalJacobian_2D, RTparams, NULL, NULL, 7, npts, 128, opts, this);
	PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");

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

// uses both 2D and 3D feature points
// for outer contour and chin region, use ONLY 2D feature points
// for other points, use 2D when depth data is not available
void evalCost(float *p, float *hx, int m, int n, void* adata) {
	//PhGUtils::message("cost func");

	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto targets_2d = recon->targets_2d;
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;
	auto w_outer = recon->w_outer;
	auto w_history = recon->w_history;
	auto meanRT = recon->meanRT;

	auto fitScale = recon->fitScale;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		// p = R * p + T
		PhGUtils::transformPoint( px, py, pz, R, T );

		// for mouth region and outer contour, use only 2D info
		if( i < 42 || i > 74 ) {
			const PhGUtils::Point3f& q = targets[i].first;

			float dx = px - q.x, dy = py - q.y, dz = pz - q.z;
			hx[i] = (dx * dx + dy * dy + dz * dz) * wpt;
		}
		else {
			if( i > 63 ) {
				wpt *= w_outer;
			}
			else
				wpt *= w_chin;

			float u, v, d;
			PhGUtils::worldToColor(px, py, pz, u, v, d);
			const PhGUtils::Point3f& q = targets_2d[i].first;

			//cout << "#" << i <<"\t" << u << ", " << v << "\t" << q.x << ", " << q.y << endl;

			float du = u - q.x, dv = v - q.y;
			hx[i] = (du * du + dv * dv) * wpt;
		}
	}
		
	// regularization terms
	int nterms = fitScale?7:6;
	for(int i=0, tidx=npts;i<nterms;i++, tidx++) {
		float diff = p[i] - meanRT[i];
		hx[tidx] = diff * diff * w_history;
	}
}

void evalJacobian(float *p, float *J, int m, int n, void *adata) {
	//PhGUtils::message("jac func");
	// J is a n-by-m matrix
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto targets_2d = recon->targets_2d;
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;
	auto w_outer = recon->w_outer;
	auto w_history = recon->w_history;
	auto meanRT = recon->meanRT;

	auto fitScale = recon->fitScale;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];

		// point p
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);

		// for mouth region and outer contour, use only 2D points
		if( i < 42 || i > 74 ) {
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

			if( fitScale ) {
				// \frac{\partial r_i}{\partial s}
				J[jidx++] = wpt * 2.0 * (rpx * rkx + rpy * rky + rpz * rkz);
			}
		}
		else {
			if( i > 63 ) {
				wpt *= w_outer;
			}
			else
				wpt *= w_chin;

			// point q
			const PhGUtils::Point3f& q = targets_2d[i].first;

			// R * p
			float rpx = px, rpy = py, rpz = pz;
			PhGUtils::rotatePoint( rpx, rpy, rpz, R );

			// s * R * p + t
			float pkx = s * rpx + tx, pky = s * rpy + ty, pkz = s * rpz + tz;

			// Jf
			float inv_z = 1.0 / pkz;
			float inv_z2 = inv_z * inv_z;
			/*
			Jf[0] = f_x * inv_z; Jf[2] = -f_x * pkx * inv_z2;
			Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
			*/

			const float f_x = 525.0, f_y = 525.0;
			float Jf[6] = {0};
			Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
			Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;

			// project p to color image plane
			float pu, pv, pd;
			PhGUtils::worldToColor(pkx, pky, pkz, pu, pv, pd);
			//cout << "#" << i <<"\t" << pu << ", " << pv << "\t" << q.x << ", " << q.y << endl;

			// residue
			float rkx = pu - q.x, rky = pv - q.y;

			// J_? * p_k
			float jpx, jpy, jpz;
			// J_f * J_? * p_k
			float jfjpx, jfjpy;

			jpx = px, jpy = py, jpz = pz;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_x}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jpx = px, jpy = py, jpz = pz;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_y}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			jpx = px, jpy = py, jpz = pz;
			PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
			jfjpx = Jf[0] * jpx + Jf[2] * jpz;
			jfjpy = Jf[4] * jpy + Jf[5] * jpz;
			// \frac{\partial r_i}{\partial \theta_z}
			J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_x}
			J[jidx++] = 2.0 * (Jf[0] * rkx) * wpt;

			// \frac{\partial r_i}{\partial \t_y}
			J[jidx++] = 2.0 * (Jf[4] * rky) * wpt;

			// \frac{\partial r_i}{\partial \t_z}
			J[jidx++] = 2.0 * (Jf[2] * rkx + Jf[5] * rky) * wpt;

			if( fitScale ) {
				// \frac{\partial r_i}{\partial s}
				float jfrpx = Jf[0] * rpx + Jf[2] * rpz;
				float jfrpy = Jf[4] * rpy + Jf[5] * rpz;
				J[jidx++] = 2.0 * (jfrpx * rkx + jfrpy * rky) * wpt;
			}
		}
	}
	
	// regularization terms
	int nterms = fitScale?7:6;
	for(int i=0, jidx=npts*nterms;i<nterms;i++) {
		float diff = p[i] - meanRT[i];
		if( diff == 0 ) diff = numeric_limits<float>::min();
		for(int j=0;j<nterms;j++) {
			if( j == i ) {
				J[jidx] = 2.0 * diff * w_history;
			}
			else J[jidx] = 0;
			jidx++;
		}
	}
	

	/*
	// write out the Jacobian
	PhGUtils::write2file("J.txt", [&](ofstream& fout){
		PhGUtils::print2DArray(J, npts+7, 7, fout);
	});
	::system("pause");
	*/
}

bool MultilinearReconstructor::fitRigidTransformationAndScale() {
	int npts = targets.size();
	
	int nparams = fitScale?7:6;

	// use levmar
	
	/*
	vector<float> errs(npts + 7);
	slevmar_chkjac(evalCost, evalJacobian, RTparams, 7, npts + 7, this, &(errs[0]));
	PhGUtils::printVector(errs);
	::system("pause");
	*/

	/*
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	//int iters = slevmar_dif(evalCost, RTparams, &(pws.meas[0]), 7, npts+7, 128, NULL, NULL, NULL, NULL, this);
	int iters = slevmar_der(evalCost, evalJacobian, RTparams, &(pws.meas[0]), 7, npts + 7, 128, opts, NULL, NULL, NULL, this);
	*/
		
	
	// use Gauss-Newton
	float opts[3] = {0.1, 1e-3, 1e-4};
	int iters = PhGUtils::GaussNewton<float>(evalCost, evalJacobian, RTparams, NULL, NULL, nparams, npts+nparams, 128, opts, this);
	

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
	

	return diff / nparams < cc;
}

bool MultilinearReconstructor::fitIdentityWeights_withPrior_ICP() {
	cout << "fitting identity weights ..." << endl;
	// to use this method, the tensor tm1 must first be updated using the rotation matrix
	int nparams = core.dim(0);	
	int npts_ICP = icpc.size();
	int npts_feature = targets.size();
	int npts = npts_ICP + npts_feature;

	Aid_ICP = PhGUtils::DenseMatrix<float>(npts*3 + nparams, nparams);
	brhs_ICP = PhGUtils::DenseVector<float>(npts*3 + nparams);

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	// ICP terms
	int ridx = 0;
	for(int i=0;i<npts_ICP;i++, ridx+=3) {
		auto bcoords = icpc[i].bcoords;
		auto v = icpc[i].v;
		int v0 = v[0] * 3, v1 = v[1] * 3, v2 = v[2] * 3;

		for(int j=0;j<nparams;j++) {
			float x = bcoords[0] * tm1RT(j, v0  ) + bcoords[1] * tm1RT(j, v1  ) + bcoords[2] * tm1RT(j, v2  );
			float y = bcoords[0] * tm1RT(j, v0+1) + bcoords[1] * tm1RT(j, v1+1) + bcoords[2] * tm1RT(j, v2+1);
			float z = bcoords[0] * tm1RT(j, v0+2) + bcoords[1] * tm1RT(j, v1+2) + bcoords[2] * tm1RT(j, v2+2);

			Aid_ICP(ridx, j) = x * w_ICP;
			Aid_ICP(ridx+1, j) = y * w_ICP;
			Aid_ICP(ridx+2, j) = z * w_ICP;
		}
	}

	// facial feature terms
	for(int i=0, ridx=npts_ICP*3;i<npts_feature;i++, ridx+=3) {
		auto v = targets[i].second * 3;
		float wpt = w_landmarks[i*3];
		for(int j=0;j<nparams;j++) {
			Aid_ICP(ridx, j) = tm1RT(j, v) * wpt;
			Aid_ICP(ridx+1, j) = tm1RT(j, v+1) * wpt;
			Aid_ICP(ridx+2, j) = tm1RT(j, v+2) * wpt;
		}
	}

	// prior terms
	for(int j=0;j<Aid.cols();j++) {
		for(int i=0, ridx=npts*3;i<nparams;i++, ridx++) {
			Aid_ICP(ridx, j) = sigma_wid_weighted(i, j);
		}
	}

	//PhGUtils::Matrix3x3f invRmat = Rmat.inv();
	

	// assemble the right hand side, fill in the upper part as usual
	// ICP terms
	for(int i=0, ridx=0;i<npts_ICP;ridx+=3, i++) {
		const PhGUtils::Point3f& q = icpc[i].q;

		brhs_ICP(ridx) = (q.x - Tvec.x) * w_ICP;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * w_ICP;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * w_ICP;
	}

	// facial feature terms
	for(int i=0, ridx=npts_ICP*3;i<npts_feature;ridx+=3, i++) {
		const PhGUtils::Point3f& q = targets[i].first;
		float wpt = w_landmarks[i*3];

		brhs_ICP(ridx) = (q.x - Tvec.x) * wpt;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * wpt;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * wpt;
	}

	// prior term
	// fill in the lower part with the mean vector of identity weights
	int ndim_id = mu_wid.size();
	for(int i=0, ridx=npts*3;i<ndim_id;i++,ridx++) {
		brhs_ICP(ridx) = mu_wid_weighted(i);
	}

	cout << "matrix and rhs assembled." << endl;

	cout << "least square" << endl;
	int rtn = leastsquare<float>(Aid_ICP, brhs_ICP);
	cout << "done." << endl;
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wid(i) - brhs_ICP(i));
		Wid(i) = brhs_ICP(i);		
		//cout << params[i] << ' ';
	}

	//cout << endl;

	cout << "identity weights fitted." << endl;

	return diff / nparams < cc;
}

void evalCost2_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tm1cRT = recon->tm1cRT;
	auto w_landmarks = recon->w_landmarks;
	auto w_prior_id_2D = recon->w_prior_id_2D;
	auto mu_wid_orig = recon->mu_wid_orig;

	// set up rotation matrix and translation vector
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float wpt = w_landmarks[vidx];

		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm1cRT(j, vidx) * p[j];
			y += tm1cRT(j, vidx+1) * p[j];
			z += tm1cRT(j, vidx+2) * p[j];
		}
		const PhGUtils::Point3f& q = targets[i].first;

		float u, v, d;
		PhGUtils::worldToColor(x + T.x, y + T.y, z + T.z, u, v, d);

		float dx = q.x - u, dy = q.y - v, dz = q.z==0?0:(q.z - d);
		hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
	}

	// regularization term
	for(int i=0, cidx=npts;i<m;i++, cidx++) {
		float diff = p[i] - mu_wid_orig(i);
		hx[cidx] = diff * diff * w_prior_id_2D;
	}
}

bool MultilinearReconstructor::fitIdentityWeights_withPrior_2D()
{
	int nparams = core.dim(0);
	vector<float> params(Wid.rawptr(), Wid.rawptr()+nparams);
	int npts = targets.size();
	vector<float> meas(npts+nparams);
	int iters = slevmar_dif(evalCost2_2D, &(params[0]), &(meas[0]), nparams, npts + nparams, 128, NULL, NULL, NULL, NULL, this);
	//cout << "finished in " << iters << " iterations." << endl;

	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wid(i) - params[i]);
		Wid(i) = params[i];		
		//cout << params[i] << ' ';
	}

	return diff / nparams < cc;
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

bool MultilinearReconstructor::fitExpressionWeights_withPrior_ICP() {
	return false;
}

void evalCost3_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor* recon = static_cast<MultilinearReconstructor*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets = recon->targets;
	int npts = targets.size();
	auto tm0cRT = recon->tm0cRT;
	auto w_landmarks = recon->w_landmarks;
	auto mu_wexp_orig = recon->mu_wexp_orig;
	auto w_prior_exp_2D = recon->w_prior_exp_2D;

	// set up rotation matrix and translation vector
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float wpt = w_landmarks[vidx];

		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm0cRT(j, vidx) * p[j];
			y += tm0cRT(j, vidx+1) * p[j];
			z += tm0cRT(j, vidx+2) * p[j];
		}
		const PhGUtils::Point3f& q = targets[i].first;

		float u, v, d;
		PhGUtils::worldToColor(x + T.x, y + T.y, z + T.z, u, v, d);

		float dx = q.x - u, dy = q.y - v, dz = q.z==0?0:(q.z - d);
		hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
	}

	// regularization term
	for(int i=0, cidx=npts;i<m;i++, cidx++) {
		float diff = p[i] - mu_wexp_orig(i);
		hx[cidx] = diff * diff * w_prior_exp_2D;
	}
}

bool MultilinearReconstructor::fitExpressionWeights_withPrior_2D()
{
	// fix both rotation and identity weights, solve for expression weights
	int nparams = core.dim(1);
	vector<float> params(Wexp.rawptr(), Wexp.rawptr() + nparams);
	int npts = targets.size();
	vector<float> meas(npts+nparams);
	int iters = slevmar_dif(evalCost3_2D, &(params[0]), &(meas[0]), nparams, npts + nparams, 128, NULL, NULL, NULL, NULL, this);

	//cout << "finished in " << iters << " iterations." << endl;

	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wexp(i) - params[i]);
		Wexp(i) = params[i];
	}

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

void MultilinearReconstructor::bindRGBDTarget(const vector<unsigned char>& colordata, const vector<unsigned char>& depthdata)
{
	targetColor = colordata;
	targetDepth = depthdata;

	// back projection to obtain 3D locations for EVERY pixel
	for(int v=0,idx=0,pidx=0;v<480;v++) {
		for(int u=0;u<640;u++,idx+=4,pidx++) {
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);
			PhGUtils::colorToWorld(
				u, v, d, 
				targetLocations[pidx].x,
				targetLocations[pidx].y,
				targetLocations[pidx].z
				);
		}
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

	if( ttp == TargetType_2D ) {
		targets_2d = pts;
		targets.resize(pts.size());
		// convert 2d to 3d
		for(int i=0;i<targets_2d.size();i++) {
			PhGUtils::colorToWorld(pts[i].first.x, pts[i].first.y, pts[i].first.z, targets[i].first.x, targets[i].first.y, targets[i].first.z);
			targets[i].second = pts[i].second;
		}
	}
	else {
		targets = pts;
	}
	int npts = targets.size();

	// compute depth mean and variance
	int validZcount = 0;
	float mu_depth = 0, sigma_depth = 0;
	for(int i=0;i<targets.size();i++) {
		float z = targets[i].first.z;
		if( z != 0 ){
			mu_depth += z;
			validZcount++;
		}
	}
	mu_depth /= validZcount;
	for(int i=0;i<targets.size();i++) {
		float z = targets[i].first.z;
		if( z != 0 ){
			float dz = z - mu_depth;
			sigma_depth += dz * dz;
		}
	}
	sigma_depth /= (validZcount-1);

	const float DEPTH_THRES = 1e-6;
	int validCount = 0;
	meanX = 0; meanY = 0; meanZ = 0;
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

		float dz = p.z - mu_depth;
		float w_depth = exp(-fabs(dz) / (sigma_depth*1e2));

		// set the landmark weights
		w_landmarks[idx] = w_landmarks[idx+1] = w_landmarks[idx+2] = (i<64 || i>74)?isValid*w_depth:isValid*w_boundary*w_depth;
		

		validCount += isValid;
	}

	meanX /= validCount;
	meanY /= validCount;
	meanZ /= validCount;

	if( updateTC ) {
		pws.meas.resize(npts + 7);
		updateComputationTensor();
		updateMatrices();

		// initialize rotation and translation

		RTparams[0] = 0;
		RTparams[1] = 0;
		RTparams[2] = 0;
		RTparams[3] = meanX;
		RTparams[4] = meanY;
		RTparams[5] = meanZ;
		RTparams[6] = fabs(targets[64].first.x - targets[74].first.x);

		// set the initial mean RT
		for(int i=0;i<7;i++) meanRT[i] = RTparams[i];
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
void MultilinearReconstructor::transformTM0()
{
	int npts = tm0.dim(1) / 3;
	// don't use the assignment operator, it actually moves the data
	//tm0cRT = tm0c;
	for(int i=0;i<tm0.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {

			tm0RT(i, vidx) = tm0(i, vidx);
			tm0RT(i, vidx+1) = tm0(i, vidx+1);
			tm0RT(i, vidx+2) = tm0(i, vidx+2);
			// store the rotated tensor in tm0cRT
			PhGUtils::rotatePoint( tm0RT(i, vidx), tm0RT(i, vidx+1), tm0RT(i, vidx+2), Rmat );
		}
	}
}

// transform TM0C with global rigid transformation
void MultilinearReconstructor::transformTM1()
{
	cout << "Transform TM1 ..." << endl;
	int npts = tm1.dim(1) / 3;
	// don't use the assignment operator, it actually moves the data
	//tm1cRT = tm1c;
	for(int i=0;i<tm1.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {

			tm1RT(i, vidx) = tm1(i, vidx);
			tm1RT(i, vidx+1) = tm1(i, vidx+1);
			tm1RT(i, vidx+2) = tm1(i, vidx+2);

			// rotation only!!!
			PhGUtils::rotatePoint( tm1RT(i, vidx), tm1RT(i, vidx+1), tm1RT(i, vidx+2), Rmat );
		}
	}
	cout << "done." << endl;
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

void MultilinearReconstructor::updateTM()
{
	tm1.modeProduct(Wid, 0, tplt);
}

void MultilinearReconstructor::updateTMwithTM0()
{
	tm0.modeProduct(Wexp, 0, tplt);
}

float MultilinearReconstructor::computeError()
{
	int npts = targets.size();
	float E = 0;
	float wsum = 0;
	for(int i=0;i<npts;i++) {
		int vidx = i * 3;
		// should change this to a fast version
		PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));

		/*p = Rmat * p + Tvec;*/		
		PhGUtils::transformPoint(p.x, p.y, p.z, Rmat, Tvec);

		E += p.distanceTo(targets[i].first) * w_landmarks[vidx];
		wsum += w_landmarks[vidx];
	}

	E /= wsum;
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

// render the mesh to the FBO
// need to call update mesh before rendering it, if the latest mesh is needed
void MultilinearReconstructor::renderMesh()
{
	dummyWgt->makeCurrent();
	fbo->bind();

#if FBO_DEBUG
	cout << (fbo->isBound()?"bounded.":"not bounded.") << endl;
	cout << (fbo->isValid()?"valid.":"invalid.") << endl;
#endif

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glPushMatrix();

	// setup viewing parameters
	glViewport(0, 0, 640, 480);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(mProj.data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixf(mMv.data());

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glShadeModel(GL_SMOOTH);

	baseMesh.drawFaceIndices();	

	glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, &(depthMap[0]));
#if FBO_DEBUG
	GLenum errcode = glGetError();
	if (errcode != GL_NO_ERROR) {
		const GLubyte *errString = gluErrorString(errcode);
		fprintf (stderr, "OpenGL Error: %s\n", errString);
	}
#endif

	glReadPixels(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, &(indexMap[0]));
#if FBO_DEBUG
	errcode = glGetError();
	if (errcode != GL_NO_ERROR) {
		const GLubyte *errString = gluErrorString(errcode);
		fprintf (stderr, "OpenGL Error: %s\n", errString);
	}
#endif

	glPopMatrix();

	glDisable(GL_CULL_FACE);

	fbo->release();
	dummyWgt->doneCurrent();

#if FBO_DEBUG
	ofstream fout("fbodepth.txt");
	PhGUtils::print2DArray(&(depthMap[0]), 480, 640, fout);
	fout.close();

	QImage img = PhGUtils::toQImage(&(indexMap[0]), 640, 480);	
	img.save("fbo.png");
#endif
}

void MultilinearReconstructor::updateMesh()
{
	for(int i=0,idx=0;i<tplt.length()/3;i++) {
		baseMesh.vertex(i).x = tmesh(idx++);
		baseMesh.vertex(i).y = tmesh(idx++);
		baseMesh.vertex(i).z = tmesh(idx++);
	}
}

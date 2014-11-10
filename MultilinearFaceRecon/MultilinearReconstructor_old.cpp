#include "MultilinearReconstructor_old.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Utils/fileutils.h"
#include "Kinect/KinectUtils.h"
#include "Math/denseblas.h"
#include "Math/Optimization.hpp"
#include "Geometry/MeshLoader.h"
#include "Geometry/MeshWriter.h"
#include "Geometry/Mesh.h"
#include "Geometry/geometryutils.hpp"
#include "Geometry/AABB.hpp"
#include "Geometry/MeshViewer.h"

#include "omp.h"

#define USELEVMAR4WEIGHTS 0
#define USE_MKL_LS 1		// use mkl least square solver
#define OUTPUT_STATS 0
#define FBO_DEBUG 0

#define RIGID_TRANS_FIT_USE_DIFF 0
#define IDENTITY_FIT_USE_DIFF 0
#define EXPRESSION_FIT_USE_DIFF 0

MultilinearReconstructor_old::MultilinearReconstructor_old(void)
{	
	culaStatus stas = culaInitialize();
	PhGUtils::culaCheckStatus(stas);

	w_prior_id = 1e-3;
	// for 25 expression dimensions
	//w_prior_exp = 1e-4;
	// for 47 expression dimensions
	w_prior_exp = 2.5e-4;

	// for outer contour and chin region, use only 2D feature point info
	w_boundary = 1e-8;
	w_chin = 1e-6;
	w_outer = 1e2;
	w_fp = 2.5;

  w_prior_exp_2D = 1.0;
  w_prior_id_2D = 5.0;
	w_boundary_2D = 1.0;

	w_history = 0.0001;

	w_ICP = 1.5;

	meanX = meanY = meanZ = 0;

	mkl_set_num_threads(8);

	loadCoreTensor();
	loadPrior();
	initializeWeights();
	createTemplateItem();
	initRigidTransformationParams();
	updateMatrices();

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
	errorThreshold = 1e-3;
	errorDiffThreshold = errorThreshold * 0.005;

	errorThreshold_ICP = 1e-5;
	errorDiffThreshold_ICP = errorThreshold * 1e-4;

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

MultilinearReconstructor_old::~MultilinearReconstructor_old(void)
{
}


void MultilinearReconstructor_old::reset()
{
	updateMatrices();
	// reset the identity weights and expression weights
	initializeWeights();

	// reset the template mesh
	createTemplateItem();

	// reset the rigid transformation parameters
	initRigidTransformationParams();
}

void MultilinearReconstructor_old::togglePrior()
{
	usePrior = !usePrior;
	PhGUtils::message((usePrior)?"Using prior":"Not using prior");
	initRigidTransformationParams();
	updateMatrices();
}


void MultilinearReconstructor_old::loadPrior()
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

void MultilinearReconstructor_old::loadCoreTensor()
{
	const string filename = "../Data/blendshape/core.bin";
	core.read(filename);
}

void MultilinearReconstructor_old::unfoldTensor()
{
	tu0 = core.unfold(0);
	tu1 = core.unfold(1);
}

void MultilinearReconstructor_old::createTemplateItem()
{
	PhGUtils::message("creating template face ...");
	tplt = tm1.modeProduct(Wid, 0);
	tmesh = tplt;
	PhGUtils::message("done.");
}

void MultilinearReconstructor_old::initializeWeights()
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
	PhGUtils::write2file("tm0_cpu.txt", [&](ostream& os){tm0.print("", os);});
	tm0RT = tm0;
	tm1 = core.modeProduct(Wexp, 1);
	PhGUtils::write2file("tm1_cpu.txt", [&](ostream& os){tm1.print("", os);});
	tm1RT = tm1;
	PhGUtils::message("done.");
}

void MultilinearReconstructor_old::initRigidTransformationParams()
{
	RTparams[0] = 0; RTparams[1] = 0; RTparams[2] = 0;
	RTparams[3] = meanX; RTparams[4] = meanY; RTparams[5] = meanZ;
	RTparams[6] = 1.0;

	R = arma::fmat(3, 3);
	R(0, 0) = 1.0, R(1, 1) = 1.0, R(2, 2) = 1.0;
	T = arma::fvec(3);

	Rmat = PhGUtils::Matrix3x3f::identity();
	Tvec = PhGUtils::Point3f::zero();
}

void MultilinearReconstructor_old::fitICP(FittingOption ops /*= FIT_ALL*/)
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
void MultilinearReconstructor_old::createFaceMask()
{
#define USE_FACE_MASK 0
#if USE_FACE_MASK
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
#else
	// uniform mask
	for(int i=0, idx=0;i<480;i++) {
		for(int j=0;j<640;j++, idx++) {
			faceMask[idx] = 1;
		}
	}
#endif
	/*
	PhGUtils::write2file("hull.txt", [&](ofstream& fout){
		PhGUtils::printVector(hull, fout);
	});
	PhGUtils::write2file("mask.txt", [&](ofstream& fout){
		PhGUtils::dump2DArray(&(faceMask[0]), 480, 640, fout);
	});
	*/
}

void MultilinearReconstructor_old::collectICPConstraints(int iter, int maxIter)
{
	const float DIST_THRES_MAX = 0.010;
	const float DIST_THRES_MIN = 0.001;
	float DIST_THRES = DIST_THRES_MAX + (DIST_THRES_MIN - DIST_THRES_MAX) * iter / (float)maxIter;
	//PhGUtils::message("Collecting ICP constraints...");
	icpc.clear();
	icpc.reserve(16384);
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
				const int wSize = 5;
				set<int> checkedFaces;

				float closestDist = numeric_limits<float>::max();
				PhGUtils::Point3i closestVerts;
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
									closestVerts.x = f.x, closestVerts.y = f.y, closestVerts.z = f.z;
									closestHit = hit1;
								}
								else if( dist2 < closestDist ) {
									closestDist = dist2;
									closestVerts.x = f.y, closestVerts.y = f.z, closestVerts.z = f.w;
									closestHit = hit2;
								}
							}
							else {
								// already checked, do nothing
							}
						}						
					}
				}
				
				// close enough to be a constraint
				if( closestDist < DIST_THRES ) {
					ICPConstraint cc;
					cc.q = q;
					cc.v = closestVerts;
					PhGUtils::computeBarycentricCoordinates(
						closestHit,
						baseMesh.vertex(closestVerts.x), baseMesh.vertex(closestVerts.y), baseMesh.vertex(closestVerts.z), 
						cc.bcoords);

					icpc.push_back(cc);
				}
			}
		}
	}

	// pick just a portion of all constraints
	
	cout << "done. ICP constraints: " << icpc.size() << endl;
	
	/*
	int maxCons = icpc.size() / 2;
	int minCons = icpc.size() / 20;

	int nsamples = minCons + (maxCons - minCons) * iter / (float)maxIter;
	std::random_shuffle(icpc.begin(), icpc.end());
	icpc.resize(nsamples);
	*/
	
	/*
	// output ICP constraints to file
	ofstream fout("icpc.txt");
	for(int i=0;i<icpc.size();i++) {
		PhGUtils::Point3f p(0, 0, 0);
		p = p + icpc[i].bcoords.x * baseMesh.vertex(icpc[i].v.x);
		p = p + icpc[i].bcoords.y * baseMesh.vertex(icpc[i].v.y);
		p = p + icpc[i].bcoords.z * baseMesh.vertex(icpc[i].v.z);
		fout << icpc[i].q << " " << p << endl;
	}
	fout.close();

	::system("pause");
	*/
}

void MultilinearReconstructor_old::collectICPConstraints_topo(int iter, int maxIter)
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
				PhGUtils::Point3i closestVerts;
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
									closestVerts.x = f.x, closestVerts.y = f.y, closestVerts.z = f.z;
									closestHit = hit1;
								}
								else if( dist2 < closestDist ) {
									closestDist = dist2;
									closestVerts.x = f.y, closestVerts.z = f.z, closestVerts.y = f.w;
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
					cc.v = closestVerts;
					PhGUtils::computeBarycentricCoordinates(
						closestHit,
						baseMesh.vertex(closestVerts.x), baseMesh.vertex(closestVerts.y), baseMesh.vertex(closestVerts.z), 
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

void MultilinearReconstructor_old::collectICPConstraints_bruteforce(int iter, int maxIter) {
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

				PhGUtils::Point3i closestVerts;
				PhGUtils::Point3f bcoords;

				float closestDist = baseMesh.findClosestPoint_bruteforce(q, closestVerts, bcoords);

				const float DIST_THRES = 0.005;				
				// close enough to be a constraint
				if( closestDist < DIST_THRES ) {
					ICPConstraint cc;
					cc.q = q;
					cc.v = closestVerts;
					cc.bcoords = bcoords;

					icpc.push_back(cc);
				}
			}
		}
	}
	
	cout << "ICP constraints: " << icpc.size() << endl;

	
	// output ICP constraints to file
	ofstream fout("icpc.txt");
	for(int i=0;i<icpc.size();i++) {
		PhGUtils::Point3f p(0, 0, 0);
		p = p + icpc[i].bcoords.x * baseMesh.vertex(icpc[i].v.x);
		p = p + icpc[i].bcoords.y * baseMesh.vertex(icpc[i].v.y);
		p = p + icpc[i].bcoords.z * baseMesh.vertex(icpc[i].v.z);
		fout << icpc[i].q << " " << p << endl;
	}
	fout.close();

	::system("pause");
}

void MultilinearReconstructor_old::fitICP_withPrior() {
	PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

	//cout << "initial guess ..." << endl;
	//PhGUtils::printArray(RTparams, 7);
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
	const int MaxIterations = 64;
	while( !converged && iters++ < MaxIterations ) {
		converged = true;

		// update mesh and render the mesh
		transformMesh();
		updateMesh();
		renderMesh();

		// collect ICP constraints
		createFaceMask();
		//PhGUtils::Timer tCons;
		//tCons.tic();
		collectICPConstraints(iters, MaxIterations);
		//collectICPConstraints_topo(iters, MaxIterations);
		//collectICPConstraints_bruteforce(iters, MaxIterations);
		//tCons.toc("constraint collection");


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
		E = computeError_ICP();
		PhGUtils::debug("iters", iters, "Error", E);

		// adaptive threshold
		bool errCC = E < (errorThreshold_ICP / (icpc.size()/5000.0));
		if( errCC ) cout << "error converged." << endl;
		converged |= errCC;
		bool derrCC = fabs(E - E0) < errorDiffThreshold_ICP;
		if( derrCC ) cout << "error difference converged." << endl;
		converged |= derrCC;
		E0 = E;
		timerOther.toc();
		//emit oneiter();
		//QApplication::processEvents();
	}

	timerTransform.tic();
	/*
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
	else */if( fitIdentity && fitExpression ) {

		tplt = core.modeProduct(Wexp, 1).modeProduct(Wid, 0);
	}
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

	ofstream fout;
	fout.open("c_wexp.txt", ios::app);
	for(int i=0;i<Wexp.length();i++) 
		fout << Wexp(i) << ' ';
	fout << endl;
	fout.close();

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

void MultilinearReconstructor_old::fit2d(FittingOption ops /*= FIT_ALL*/)
{
	switch( ops ) {
	case FIT_POSE:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = false;
      fitScale = true;
      fit2d_pose();
			break;
		}
	case FIT_IDENTITY:
		{
			fitPose = false;
			fitIdentity = true;
			fitExpression = false;
      fit2d_identity();
			break;
		}
	case FIT_EXPRESSION:
		{
			fitPose = false;
			fitIdentity = false;
			fitExpression = true;
      fit2d_expression();
			break;
		}
	case FIT_POSE_AND_IDENTITY:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = false;
      fitScale = false;
      //fit2d_poseAndIdentity();
      fit2d_pose_identity_camera();
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitPose = true;
			fitIdentity = false;
			fitExpression = true;
      fitScale = false;
      fit2d_poseAndExpression();
			break;
		}
	case FIT_ALL:
		{
			fitPose = true;
			fitIdentity = true;
			fitExpression = true;
      fit2d_all();
			break;
		}
	}
	frameCounter++;
}

void MultilinearReconstructor_old::fit2d_pose() {
  cout << "fit with prior using 2D feature points" << endl;

  //init();

  if (targets.empty())
  {
    PhGUtils::error("No target set!");
    return;
  }
  PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

  timerTotal.tic();
  int iters = 0;
  float E0 = 0;
  bool converged = false;
  //while (!converged && iters++ < MAXITERS) {
    converged = true;

    timerRT.tic();
    converged &= fitRigidTransformationAndScale_2D();
    timerRT.toc();

    E = computeError_2D();
    PhGUtils::info("iters", iters, "Error", E);

    converged |= E < errorThreshold;
    converged |= fabs(E - E0) < errorDiffThreshold;
    E0 = E;
  //}

  if (useHistory) {
    // post process, impose a moving average for pose
    RTHistory.push_back(vector<float>(RTparams, RTparams + 7));
    if (RTHistory.size() > historyLength) RTHistory.pop_front();
    vector<float> meanRT = computeWeightedMeanPose();
    // copy back the mean pose
    for (int i = 0; i<7; i++) RTparams[i] = meanRT[i];
  }

  timerTransform.tic();
  transformMesh();
  timerTransform.toc();

  timerTotal.toc();

#if OUTPUT_STATS
  cout << "Total iterations = " << iters << endl;
  PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed() * 1000) + " ms.");
  PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed() * 1000) + " ms.");
#endif
}

void MultilinearReconstructor_old::fit2d_identity() {
  cout << "fit with prior using 2D feature points" << endl;

  //init();

  if (targets.empty())
  {
    PhGUtils::error("No target set!");
    return;
  }
  PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

  timerTotal.tic();

  // apply the new global transformation to tm1c
  // because tm1c is required in fitting identity weights
  timerOther.tic();
  transformTM1C();
  timerOther.toc();

  int iters = 0;
  float E0 = 0;
  bool converged = false;
  while (!converged && iters++ < MAXITERS) {
    converged = true;

    timerID.tic();
    converged &= fitIdentityWeights_withPrior_2D();
    timerID.toc();

    E = computeError_2D();
    PhGUtils::info("iters", iters, "Error", E);

    converged |= E < errorThreshold;
    converged |= fabs(E - E0) < errorDiffThreshold;
    E0 = E;
    timerOther.toc();
  }

  timerTransform.tic();
  tplt = tm1.modeProduct(Wid, 0);
  
  timerTransform.tic();
  //PhGUtils::debug("R", Rmat);
  //PhGUtils::debug("T", Tvec);
  transformMesh();
  timerTransform.toc();
  //emit oneiter();
  timerTotal.toc();

#if OUTPUT_STATS
  cout << "Total iterations = " << iters << endl;
  PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed() * 1000) + " ms.");
  PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed() * 1000) + " ms.");
#endif
}

void MultilinearReconstructor_old::fit2d_expression() {

}

void MultilinearReconstructor_old::fit2d_poseAndIdentity() {
  if (targets.empty())
  {
    PhGUtils::error("No target set!");
    return;
  }
  PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

  timerTotal.tic();
  int iters = 0;
  float E0 = 0;
  bool converged = false;
  while (!converged && iters++ < MAXITERS) {
    converged = true;

    timerRT.tic();
    converged &= fitRigidTransformationAndScale_2D();
    timerRT.toc();

    // apply the new global transformation to tm1c
    // because tm1c is required in fitting identity weights
    timerOther.tic();
    transformTM1C();
    timerOther.toc();

    timerID.tic();
    converged &= fitIdentityWeights_withPrior_2D();
    timerID.toc();

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    tm0c = corec.modeProduct(Wid, 0);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();
   
    // compute tmc from the new tm1c or new tm0c
    timerOther.tic();
    updateTMC();
    timerOther.toc();

    timerOther.tic();

    E = computeError_2D();
    PhGUtils::info("iters", iters, "Error", E);

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
  tplt = tm1.modeProduct(Wid, 0);
  timerTransform.toc();

  if (useHistory) {
    // post process, impose a moving average for pose
    RTHistory.push_back(vector<float>(RTparams, RTparams + 7));
    if (RTHistory.size() > historyLength) RTHistory.pop_front();
    vector<float> meanRT = computeWeightedMeanPose();
    // copy back the mean pose
    for (int i = 0; i<7; i++) RTparams[i] = meanRT[i];
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
  PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed() * 1000) + " ms.");
  PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed() * 1000) + " ms.");
#endif
}

void MultilinearReconstructor_old::fit2d_pose_identity_camera() {
  if (targets.empty())
  {
    PhGUtils::error("No target set!");
    return;
  }
  PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

  timerTotal.tic();
  int iters = 0;
  float E0 = 0;
  bool converged = false;
  while (!converged && iters++ < MAXITERS) {
    converged = true;

    timerRT.tic();
    converged &= fitRigidTransformationAndScale_2D();
    timerRT.toc();

    // apply the new global transformation to tm1c
    // because tm1c is required in fitting identity weights
    timerOther.tic();
    transformTM1C();
    timerOther.toc();

    timerID.tic();
    converged &= fitIdentityWeights_withPrior_2D();
    timerID.toc();

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    tm0c = corec.modeProduct(Wid, 0);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    // compute tmc from the new tm1c or new tm0c
    timerOther.tic();
    updateTMC();
    timerOther.toc();

    /// optimize for camera parameters
    converged &= fitCameraParameters_2D();

    timerOther.tic();
    E = computeError_2D();
    PhGUtils::info("iters", iters, "Error", E);

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

  timerOther.tic();
  // update TM0
  tm0 = core.modeProduct(Wid, 0);
  timerOther.toc();

  timerTransform.tic();
  tplt = tm1.modeProduct(Wid, 0);
  timerTransform.toc();

  if (useHistory) {
    // post process, impose a moving average for pose
    RTHistory.push_back(vector<float>(RTparams, RTparams + 7));
    if (RTHistory.size() > historyLength) RTHistory.pop_front();
    vector<float> meanRT = computeWeightedMeanPose();
    // copy back the mean pose
    for (int i = 0; i<7; i++) RTparams[i] = meanRT[i];
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
  PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wid fitting = " + PhGUtils::toString(timerID.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed() * 1000) + " ms.");
  PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed() * 1000) + " ms.");
#endif
}

void MultilinearReconstructor_old::fit2d_poseAndExpression() {
  if (targets.empty())
  {
    PhGUtils::error("No target set!");
    return;
  }
  PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

  timerTotal.tic();
  int iters = 0;
  float E0 = 0;
  bool converged = false;
  while (!converged && iters++ < MAXITERS) {
    converged = true;

    timerRT.tic();
    converged &= fitRigidTransformationAndScale_2D();
    timerRT.toc();

    //cout << "fitting expression weights ..." << endl;
    timerOther.tic();
    // apply the global transformation to tm0c
    // because tm0c is required in fitting expression weights
    transformTM0C();
    timerOther.toc();

    timerExp.tic();
    converged &= fitExpressionWeights_withPrior_2D();
    timerExp.toc();

    // compute tmc from the new tm1c or new tm0c
    timerOther.tic();
    updateTMCwithTM0C();
    //updateTMC();
    timerOther.toc();

    E = computeError_2D();
    //PhGUtils::info("iters", iters, "Error", E);

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
  tplt = tm0.modeProduct(Wexp, 0);
  timerTransform.toc();

  if (useHistory) {
    // post process, impose a moving average for pose
    RTHistory.push_back(vector<float>(RTparams, RTparams + 7));
    if (RTHistory.size() > historyLength) RTHistory.pop_front();
    vector<float> meanRT = computeWeightedMeanPose();
    // copy back the mean pose
    for (int i = 0; i<7; i++) RTparams[i] = meanRT[i];
  }

  timerTransform.tic();
  //PhGUtils::debug("R", Rmat);
  //PhGUtils::debug("T", Tvec);
  transformMesh();
  timerTransform.toc();
  //emit oneiter();
  timerTotal.toc();

#if 0
  cout << "Total iterations = " << iters << endl;
  PhGUtils::message("Time cost for pose fitting = " + PhGUtils::toString(timerRT.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for wexp fitting = " + PhGUtils::toString(timerExp.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for tensor transformation = " + PhGUtils::toString(timerTransform.elapsed() * 1000) + " ms.");
  PhGUtils::message("Time cost for other computation = " + PhGUtils::toString(timerOther.elapsed() * 1000) + " ms.");
  PhGUtils::message("Total time cost for reconstruction = " + PhGUtils::toString(timerTotal.elapsed() * 1000) + " ms.");
#endif
}

void MultilinearReconstructor_old::fit2d_all() {

}

void MultilinearReconstructor_old::fit2d_withPrior() {
	cout << "fit with prior using 2D feature points" << endl;

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
      cout << "fitting expression weights ..." << endl;
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
		PhGUtils::info("iters", iters, "Error", E);

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

void MultilinearReconstructor_old::fit( MultilinearReconstructor_old::FittingOption ops )
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

void MultilinearReconstructor_old::fit_withPrior() {
	//cout << "fit with prior" << endl;

	//init();

	if(targets.empty())
	{
		PhGUtils::error("No target set!");
		return;
	}
	PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;
	
#ifdef _DEBUG
	PhGUtils::printArray(RTparams, 7);
	Wid.print("Wid");
	Wexp.print("Wexp");
#endif

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
		PhGUtils::debug("iters", iters, "Error", E);

#ifdef _DEBUG
		PhGUtils::printArray(RTparams, 7);
		Wid.print("Wid");
#endif

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

vector<float> MultilinearReconstructor_old::computeWeightedMeanPose() {
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
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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

	float w_fp_scale = npts_ICP / 1000.0;

	auto fitScale = recon->fitScale;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	// ICP terms
	for(int i=0;i<npts_ICP;i++) {
		auto v = icpc[i].v;
		auto bcoords = icpc[i].bcoords;

		int vidx[3];
		vidx[0] = v.x*3; vidx[1] = v.y*3; vidx[2] = v.z*3;

		PhGUtils::Point3f p;
		p.x += bcoords.x * tplt(vidx[0]); p.y += bcoords.x * tplt(vidx[0]+1); p.z += bcoords.x * tplt(vidx[0]+2);
		p.x += bcoords.y * tplt(vidx[1]); p.y += bcoords.y * tplt(vidx[1]+1); p.z += bcoords.y * tplt(vidx[1]+2);
		p.x += bcoords.z * tplt(vidx[2]); p.y += bcoords.z * tplt(vidx[2]+1); p.z += bcoords.z * tplt(vidx[2]+2);

		const PhGUtils::Point3f& q = icpc[i].q;

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, R, T );

		float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
		hx[i] = (dx * dx + dy * dy + dz * dz) * w_ICP;
	}

	// facial feature term
	for(int i=0, hxidx = npts_ICP;i<npts_feature;i++, hxidx++) {
		int vidx = fp[i].second * 3;
		float wpt = w_landmarks[i*3] * w_fp_scale;
		
		float wpt0 = wpt;

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
		//cout << "#" << i << "\t" << wpt0 << "\t" << wpt << endl;
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
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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
	float w_fp_scale = npts_ICP / 1000.0;

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
		vidx[0] = v.x*3; vidx[1] = v.y*3; vidx[2] = v.z*3;

		// point p
		PhGUtils::Point3f p;
		p.x += bcoords.x * tplt(vidx[0]); p.y += bcoords.x * tplt(vidx[0]+1); p.z += bcoords.x * tplt(vidx[0]+2);
		p.x += bcoords.y * tplt(vidx[1]); p.y += bcoords.y * tplt(vidx[1]+1); p.z += bcoords.y * tplt(vidx[1]+2);
		p.x += bcoords.z * tplt(vidx[2]); p.y += bcoords.z * tplt(vidx[2]+1); p.z += bcoords.z * tplt(vidx[2]+2);

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
		float wpt = w_landmarks[i*3] * w_fp_scale;

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

bool MultilinearReconstructor_old::fitRigidTransformationAndScale_ICP() {
	int nparams = fitScale?7:6;
	int npts = icpc.size() + targets.size() + nparams;

	/*
	vector<float> meas(npts);
	// use levmar
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	//int iters = slevmar_dif(evalCost_ICP, RTparams, &(meas[0]), nparams, npts, 128, NULL, NULL, NULL, NULL, this);
	int iters = slevmar_der(evalCost_ICP, evalJacobian_ICP, RTparams, &(meas[0]), nparams, npts, 128, opts, NULL, NULL, NULL, this);
	*/
	
	float RTparams0[7];
	for(int i=0;i<7;i++) { RTparams0[i] = RTparams[i]; }

	// use Gauss-Newton
	float opts[3] = {0.125, 1e-3, 1e-4};
	int iters = PhGUtils::GaussNewton<float>(evalCost_ICP, evalJacobian_ICP, RTparams, NULL, NULL, nparams, npts, 128, opts, this);
	

	//PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");
	
	// set up the matrix and translation vector
	Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]) * RTparams[6];
	float diff = 0;
	for(int i=0;i<7;i++) { diff += RTparams[i] - RTparams0[i]; }
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			//diff += fabs(R(i, j) - Rmat(i, j));
			R(i, j) = Rmat(i, j);			
		}
	}

	Tvec.x = RTparams[3], Tvec.y = RTparams[4], Tvec.z = RTparams[5];
	//diff += fabs(Tvec.x - T(0)) + fabs(Tvec.y - T(1)) + fabs(Tvec.z - T(2));
	T(0) = RTparams[3], T(1) = RTparams[4], T(2) = RTparams[5];

	/*
	cout << R << endl;
	cout << T << endl;
	*/

	bool converged = diff / nparams < cc || iters == 0;
	if( converged ) cout << "rigid transformation converged." << endl;
	return converged;
}

void evalCost_pose_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
  rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

	auto targets_2d = recon->targets_2d;
	int npts = targets_2d.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;

	// set up rotation matrix and translation vector
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
                             0, recon->camparams.fy, recon->camparams.cy,
                             0, 0, 1);

	// apply the new global transformation
	for(int i=0, vidx=0;i<npts;i++) {
		//PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
		float wpt = w_landmarks[vidx];

    // exclude mouth region and chin region
    if (recon->fitExpression) {
      if (i >= 48 || (i >= 4 && i <= 12))
        wpt = 0.0;
    }

		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		const PhGUtils::Point3f& q = targets_2d[i].first;

		// PhGUtils::Point3f pp = R * p + T;
		PhGUtils::transformPoint( px, py, pz, R, T );
    //cout << px << " " << py << " " << pz << endl;

		// Projection to image plane
		float u, v;
    PhGUtils::projectPoint(u, v, px, py, pz, Mproj);

		//cout << i << "\t" << u << ", " << v << endl;

		float dx = q.x - u, dy = q.y - v;
		//cout << i << "\t" << dx << ", " << dy << ", " << dz << endl;
    //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
    hx[i] = sqrt(dx * dx + dy * dy) * wpt;
	}
}

void evalJacobian_pose_2D(float *p, float *J, int m, int n, void *adata) {
	// J is a n-by-m matrix
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

	float rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

  auto targets_2d = recon->targets_2d;
  int npts = targets_2d.size();
	auto tmc = recon->tmc;

	auto w_landmarks = recon->w_landmarks;
	auto w_chin = recon->w_chin;

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
    0, recon->camparams.fy, recon->camparams.cy,
    0, 0, 1);

	//cout << Jx << endl;
	//cout << Jy << endl;
	//cout << Jz << endl;

	// for Jacobian of projection/viewport transformation
  const float f_x = recon->camparams.fx, f_y = recon->camparams.fy;
	float Jf[6] = {0};

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<npts;i++) {
		float wpt = w_landmarks[vidx];

    // exclude mouth region and chin region
    if (recon->fitExpression) {
      if (i >= 48 || (i >= 4 && i <= 12))
        wpt = 0.0;
    }

		// point p
		float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
		//cout << px << ", " << py << ", " << pz << endl;

		// point q
		const PhGUtils::Point3f& q = targets_2d[i].first;

		// R * p
		float Rpx = px, Rpy = py, Rpz = pz;
		PhGUtils::rotatePoint( Rpx, Rpy, Rpz, R );

		// P = R * p + t
		float Pkx = Rpx + tx, Pky = Rpy + ty, Pkz = Rpz + tz;

		// Jf
		float inv_z = 1.0 / Pkz;
		float inv_z2 = inv_z * inv_z;
		
		Jf[0] = f_x * inv_z; Jf[2] = -f_x * Pkx * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * Pky * inv_z2;
		
    /*
		Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
    */

		// project p to color image plane
		float pu, pv;
    PhGUtils::projectPoint(pu, pv, Pkx, Pky, Pkz, Mproj);

		// residue
		float rkx = pu - q.x, rky = pv - q.y;
    double rk = sqrt(rkx*rkx + rky*rky);
    double inv_rk = 1.0 / rk;
    // normalize it
    rkx *= inv_rk; rky *= inv_rk;

		// J_? * p_k
		float jpx, jpy, jpz;
		// J_f * J_? * p_k
		float jfjpx, jfjpy;

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		/*
		cout << "jf\t";
		PhGUtils::printArray(Jf, 9);
		cout << "j\t" << jpx << ", " << jpy << ", "  << jpz << endl;
		cout << "jj\t" << jfjpx << ", " << jfjpy << ", "  << jfjpz << endl;
		*/

		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		// \frac{\partial r_i}{\partial \theta_y}
    J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

		jpx = px, jpy = py, jpz = pz;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		jfjpx = Jf[0] * jpx + Jf[2] * jpz;
		jfjpy = Jf[4] * jpy + Jf[5] * jpz;
		// \frac{\partial r_i}{\partial \theta_z}
    J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

		// \frac{\partial r_i}{\partial \t_x}
    J[jidx++] = (Jf[0] * rkx) * wpt;

		// \frac{\partial r_i}{\partial \t_y}
    J[jidx++] = (Jf[4] * rky) * wpt;

		// \frac{\partial r_i}{\partial \t_z}
    J[jidx++] = (Jf[2] * rkx + Jf[5] * rky) * wpt;
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

bool MultilinearReconstructor_old::fitCameraParameters_2D() {
  int npts = targets.size();
  /// compute the focal length analytically
  double numer = 0.0, denom = 0.0;
  double p0 = camparams.cx, q0 = camparams.cy;
  for (int i = 0, vidx = 0; i < npts; ++i) {
    float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
    float pi = targets_2d[i].first.x, qi = targets_2d[i].first.y;
    PhGUtils::transformPoint(px, py, pz, Rmat, Tvec);
    float xz = px / pz;
    float yz = py / pz;
    numer += yz * (q0 - qi) - xz*(p0 - pi);
    denom += (yz*yz+xz*xz);
  }
  double nf = -numer / denom;
  cout << nf << endl;
  camparams.fx = -nf;
  camparams.fy = nf;
  return true;
}

bool MultilinearReconstructor_old::fitRigidTransformationAndScale_2D() {
	int npts = targets.size();
	// use levmar

	/*
	vector<float> errs(npts);
	slevmar_chkjac(evalCost_2D, evalJacobian_2D, RTparams, 7, npts, this, &(errs[0]));
	PhGUtils::printVector(errs);
	::system("pause");
	*/

#if 0
  for (int i = 0, vidx = 0; i < npts; ++i) {
    float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
    cout << "(" << px << ", " << py << ", " << pz << ") ";
  }
  cout << endl;
#endif

  //cout << "before" << endl;
  //cout << R << endl;
  //cout << T << endl;

	vector<float> meas(npts, 0.0);

  int nparams = 6;
  //cout << "nparams = " << nparams << endl;
  //cout << "before: "
  //  << RTparams[0] << ", "
  //  << RTparams[1] << ", "
  //  << RTparams[2] << ", "
  //  << RTparams[3] << ", "
  //  << RTparams[4] << ", "
  //  << RTparams[5] << endl;
  float lminfo[10];
  string lmStopReason[8] = {
    "",
    "stopped by small gradient J^T e",
    "stopped by small Dp",
    "stopped by itmax",
    "singular matrix.Restart from current p with increased mu",
    "no further error reduction is possible.Restart with increased mu",
    "stopped by small ||e||_2",
    "stopped by invalid(i.e.NaN or Inf) \"func\" values.This is a user error"
  };
  string infostr[10] = {
    "||e||_2 at initial p",
    "||e||_2",
    "||J^T e||_inf",
    "||Dp||_2",
    "mu / max[J^T J]_ii",
    "# iterations",
    "reason for termination:",
    "# function evaluations",
    "# Jacobian evaluations",
    "# linear systems solved"
  };

#if RIGID_TRANS_FIT_USE_DIFF
  float opts[5] = { 1e-3, 1e-9, 1e-9, 1e-9, 1e-6 }; // tau, eps1, eps2, eps3, delta
  int iters = slevmar_dif(evalCost_pose_2D, RTparams, &(meas[0]), nparams, npts, 128, NULL, lminfo, NULL, NULL, this);
#else
  float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9};  // tau, eps1, eps2, eps3
  //vector<float> errorvec(npts);
  //slevmar_chkjac(evalCost_pose_2D, evalJacobian_pose_2D, RTparams, nparams, npts, this, &(errorvec[0]));
  //PhGUtils::message("error vec");
  //PhGUtils::printArray(&errorvec[0], npts);
	int iters = slevmar_der(evalCost_pose_2D, evalJacobian_pose_2D, RTparams, &(meas[0]), nparams, npts, 128, opts, NULL, NULL, NULL, this);
#endif
#if 0
  for (int i = 0; i < 10; ++i) {
    if (i == 6)
      cout << infostr[i] << ": " << lmStopReason[(int)lminfo[i]] << endl;
    else
      cout << infostr[i] << ": " << lminfo[i] << endl;
  }
  cout << endl;
#endif

	//PhGUtils::message("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");
  //cout << "after: "
  //     << RTparams[0] << ", "
  //     << RTparams[1] << ", "
  //     << RTparams[2] << ", "
  //     << RTparams[3] << ", "
  //     << RTparams[4] << ", "
  //     << RTparams[5] << endl;

	// set up the matrix and translation vector
	Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]);
	float diff = 0;
	for(int i=0;i<3;i++) {
		for(int j=0;j<3;j++) {
			diff = max(diff, fabs(R(i, j) - Rmat(i, j)));
			R(i, j) = Rmat(i, j);			
		}
	}

	Tvec.x = RTparams[3], Tvec.y = RTparams[4], Tvec.z = RTparams[5];
  diff = max(diff, fabs(Tvec.x - T(0)));
  diff = max(diff, fabs(Tvec.y - T(1)));
  diff = max(diff, fabs(Tvec.z - T(2)));
	T(0) = RTparams[3], T(1) = RTparams[4], T(2) = RTparams[5];

 // cout << "after" << endl;
	//cout << R << endl;
	//cout << T << endl;
	return diff < 2.5e-6;
}

// uses both 2D and 3D feature points
// for outer contour and chin region, use ONLY 2D feature points
// for other points, use 2D when depth data is not available
void evalCost(float *p, float *hx, int m, int n, void* adata) {
	//PhGUtils::message("cost func");

	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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
		//if( diff == 0 ) diff = numeric_limits<float>::min();
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

bool MultilinearReconstructor_old::fitRigidTransformationAndScale() {
	int npts = targets.size();
	
	int nparams = fitScale?7:6;

	// use levmar
	
	/*
	vector<float> errs(npts + 7);
	slevmar_chkjac(evalCost, evalJacobian, RTparams, nparams, npts + nparams, this, &(errs[0]));
	PhGUtils::printVector(errs);
	::system("pause");
	*/
	
	/*
	float opts[4] = {1e-3, 1e-9, 1e-9, 1e-9};
	int iters = slevmar_dif(evalCost, RTparams, &(pws.meas[0]), nparams, npts+nparams, 128, NULL, NULL, NULL, NULL, this);
	//int iters = slevmar_der(evalCost, evalJacobian, RTparams, &(pws.meas[0]), nparams, npts + nparams, 128, opts, NULL, NULL, NULL, this);
	*/

	// use Gauss-Newton
	float opts[3] = {0.1, 1e-3, 1e-4};
	int iters = PhGUtils::GaussNewton<float>(evalCost, evalJacobian, RTparams, NULL, NULL, nparams, npts+nparams, 128, opts, this);
	PhGUtils::debugmessage("rigid fitting finished in " + PhGUtils::toString(iters) + " iterations.");

	// set up the matrix and translation vector
	Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]);
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

bool MultilinearReconstructor_old::fitIdentityWeights_withPrior_ICP() {
	//cout << "fitting identity weights ..." << endl;
	// to use this method, the tensor tm1 must first be updated using the rotation matrix
	int nparams = core.dim(0);	
	int npts_ICP = icpc.size();
	int npts_feature = targets.size();
	int npts = npts_ICP + npts_feature;

	Aid_ICP = PhGUtils::DenseMatrix<double>(npts*3 + nparams, nparams);
	brhs_ICP = PhGUtils::DenseVector<double>(npts*3 + nparams);	

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	// ICP terms
	int ridx = 0;
	#pragma omp parallel for
	for(int i=0;i<npts_ICP;i++) {
		auto bcoords = icpc[i].bcoords;
		auto v = icpc[i].v;
		int v0 = v.x * 3, v1 = v.y * 3, v2 = v.z * 3;

		for(int j=0, ridx=i*3;j<nparams;j++) {
			float x = 0, y = 0, z = 0;
			x += bcoords.x * tm1RT(j, v0  ); y += bcoords.x * tm1RT(j, v0+1); z += bcoords.x * tm1RT(j, v0+2); 
			x += bcoords.y * tm1RT(j, v1  ); y += bcoords.y * tm1RT(j, v1+1); z += bcoords.y * tm1RT(j, v1+2); 
			x += bcoords.z * tm1RT(j, v2  ); y += bcoords.z * tm1RT(j, v2+1); z += bcoords.z * tm1RT(j, v2+2);

			Aid_ICP(ridx, j) = x * w_ICP;
			Aid_ICP(ridx+1, j) = y * w_ICP;
			Aid_ICP(ridx+2, j) = z * w_ICP;
		}
	}

	float w_fp_scale = npts_ICP / 1000.0;
	// facial feature terms
	for(int i=0, ridx=npts_ICP*3;i<npts_feature;i++, ridx+=3) {
		auto v = targets[i].second * 3;
		float wpt = w_landmarks[i*3] * w_fp_scale;
		for(int j=0;j<nparams;j++) {
			Aid_ICP(ridx, j) = tm1RT(j, v) * wpt;
			Aid_ICP(ridx+1, j) = tm1RT(j, v+1) * wpt;
			Aid_ICP(ridx+2, j) = tm1RT(j, v+2) * wpt;
		}
	}

	float w_prior_scale = npts_ICP / 2000.0;
	// prior terms
	for(int j=0;j<Aid_ICP.cols();j++) {
		for(int i=0, ridx=npts*3;i<nparams;i++, ridx++) {
			Aid_ICP(ridx, j) = sigma_wid_weighted(i, j) * w_prior_scale;
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
		float wpt = w_landmarks[i*3] * w_fp_scale;

		brhs_ICP(ridx) = (q.x - Tvec.x) * wpt;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * wpt;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * wpt;
	}

	// prior term
	// fill in the lower part with the mean vector of identity weights
	int ndim_id = mu_wid.size();
	for(int i=0, ridx=npts*3;i<ndim_id;i++,ridx++) {
		brhs_ICP(ridx) = mu_wid_weighted(i) * w_prior_scale;
	}

	//ofstream fout("Aid.txt");
	//PhGUtils::print2DArray(Aid_ICP.ptr(), Aid_ICP.cols(), Aid_ICP.rows(), fout);
	//fout.close();
	//cout << "matrix and rhs assembled." << endl;
	//::system("pause");

	//cout << "least square" << endl;
#if 0
	int rtn = leastsquare<float>(Aid_ICP, brhs_ICP);
#else
	PhGUtils::DenseMatrix<double> AtA = PhGUtils::DenseMatrix<double>(nparams, nparams);
	PhGUtils::DenseVector<double> Atb = PhGUtils::DenseVector<double>(nparams);
	int rtn = leastsquare_normalmat(Aid_ICP, brhs_ICP, AtA, Atb);
#endif
	//cout << "done." << endl;
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
#if 0
		diff += fabs(Wid(i) - brhs_ICP(i));
		Wid(i) = brhs_ICP(i);		
#else
		//cout << Wid(i) << '\t' << Atb(i) << endl;
		diff += fabs(Wid(i) - Atb(i));
		Wid(i) = Atb(i);	
#endif
	}
	//cout << endl;
	//::system("pause");
	//cout << "identity weights fitted." << endl;

	bool converged = diff / nparams < cc;
	if( converged ) cout << "identity weights converged." << endl;
	return converged;
}

void evalCost_identity_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	auto targets_2d = recon->targets_2d;
	int npts = targets_2d.size();
	auto tm1cRT = recon->tm1cRT;
	auto w_landmarks = recon->w_landmarks;
	auto w_prior_id_2D = recon->w_prior_id_2D;
	auto mu_wid_orig = recon->mu_wid_orig;

	// set up rotation matrix and translation vector
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
    0, recon->camparams.fy, recon->camparams.cy,
    0, 0, 1);

  //cout << "npts = " << npts << endl;
	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float wpt = w_landmarks[vidx];

		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm1cRT(j, vidx) * p[j];
			y += tm1cRT(j, vidx+1) * p[j];
			z += tm1cRT(j, vidx+2) * p[j];
		}
		const PhGUtils::Point3f& q = targets_2d[i].first;

    float u, v;
    PhGUtils::projectPoint(u, v, x + T.x, y + T.y, z + T.z, Mproj);

		float dx = q.x - u, dy = q.y - v;
    //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
    hx[i] = sqrt(dx*dx+dy*dy);
    //cout << hx[i] << ", ";
	}

	// regularization term
  arma::fvec wmmu(m);
  for (int i = 0; i < m; ++i) {
		wmmu(i) = p[i] - mu_wid_orig(i);
	}
  hx[n-1] = sqrt(arma::dot(wmmu, recon->sigma_wid * wmmu)) * w_prior_id_2D;
  //cout << hx[n] << endl;
}

void evalJacobian_identity_2D(float *p, float *J, int m, int n, void* adata) {
  MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

  float rx, ry, rz, tx, ty, tz;
  rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

  auto targets_2d = recon->targets_2d;
  int npts = targets_2d.size();
  auto tm1cRT = recon->tm1cRT;
  auto w_landmarks = recon->w_landmarks;
  auto w_prior_id_2D = recon->w_prior_id_2D;
  auto mu_wid_orig = recon->mu_wid_orig;

  // set up rotation matrix and translation vector
  const PhGUtils::Point3f& T = recon->Tvec;
  const PhGUtils::Matrix3x3f& R = recon->Rmat;
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
    0, recon->camparams.fy, recon->camparams.cy,
    0, 0, 1);

  //cout << "hx: ";
  float Jf[6] = { 0 };
  float f_x = recon->camparams.fx, f_y = recon->camparams.fy;

  int jidx = 0;
  for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
    float wpt = w_landmarks[vidx];

    // Pk
    float Px = 0, Py = 0, Pz = 0;
    for (int j = 0; j < m; j++) {
      Px += tm1cRT(j, vidx) * p[j];
      Py += tm1cRT(j, vidx + 1) * p[j];
      Pz += tm1cRT(j, vidx + 2) * p[j];
    }
    Px += T.x; Py += T.y; Pz += T.z;

    const PhGUtils::Point3f& q = targets_2d[i].first;    
    float u, v;
    PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

    float rkx = u - q.x, rky = v - q.y;

    double rk = sqrt(rkx*rkx + rky*rky);
    double inv_rk = 1.0 / rk;
    rkx *= inv_rk; rky *= inv_rk;

    // Jf
    float inv_z = 1.0 / Pz;
    float inv_z2 = inv_z * inv_z;

    Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
    Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

    for (int j = 0; j < m; ++j) {
      // R * Jm * e_j
      float jmex = tm1cRT(j, vidx), jmey = tm1cRT(j, vidx + 1), jmez = tm1cRT(j, vidx + 2);

      // Jf * R * Jm * e_j
      float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
      float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
      // data term and prior term
      J[jidx++] = (jfjmx * rkx + jfjmy * rky);
    }
  }

  // regularization term
  arma::fvec wmmu(m);
  for (int j = 0; j < m; ++j) {
    wmmu(j) = p[j] - mu_wid_orig(j);
  }
  arma::fvec Jprior = recon->sigma_wid * wmmu;
  float Eprior = sqrt(arma::dot(wmmu, Jprior));
  float inv_Eprior = 1.0 / Eprior;

  for (int j = 0; j < m; ++j) {
    J[jidx++] = inv_Eprior * Jprior(j) * w_prior_id_2D;
  }
}

bool MultilinearReconstructor_old::fitIdentityWeights_withPrior_2D()
{
	int nparams = core.dim(0);
	vector<float> params(Wid.rawptr(), Wid.rawptr()+nparams);
	int npts = targets.size();
	vector<float> meas(npts+1, 0.0);
#if IDENTITY_FIT_USE_DIFF
  float opts[5] = { 1e-3, 1e-9, 1e-9, 1e-9, 1e-6 };
  int iters = slevmar_dif(evalCost_identity_2D, &(params[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, this);
#else
  // evaluate the jacobian
  vector<float> errorvec(npts+1);
  slevmar_chkjac(evalCost_identity_2D, evalJacobian_identity_2D, &(params[0]), nparams, npts + 1, this, &(errorvec[0]));
  PhGUtils::message("error vec[identity]");
  PhGUtils::printArray(&(errorvec[0]), npts + 1);

  float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9 };
  int iters = slevmar_der(evalCost_identity_2D, evalJacobian_identity_2D, &(params[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, this);
#endif
	//cout << "identity fitting finished in " << iters << " iterations." << endl;

	//debug("rtn", rtn);
	float diff = 0.0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff = max(diff, fabs(Wid(i) - params[i]) / (fabs(Wid(i))+1e-12f));
		Wid(i) = params[i];		
		//cout << params[i] << ' ';
	}
  //cout << endl;

  cout << "diff = " << diff  << endl;

	return diff < 1e-4;
}

void evalCost2(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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

bool MultilinearReconstructor_old::fitIdentityWeights_withPrior()
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

	int rtn = PhGUtils::leastsquare<float>(Aid, brhs);
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

bool MultilinearReconstructor_old::fitExpressionWeights_withPrior_ICP() {
	//cout << "fitting identity weights ..." << endl;
	// to use this method, the tensor tm1 must first be updated using the rotation matrix
	int nparams = core.dim(1);
	int npts_ICP = icpc.size();
	int npts_feature = targets.size();
	int npts = npts_ICP + npts_feature;

	Aexp_ICP = PhGUtils::DenseMatrix<double>(npts*3 + nparams, nparams);
	brhs_ICP = PhGUtils::DenseVector<double>(npts*3 + nparams);

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	// ICP terms
	#pragma omp parallel for
	for(int i=0;i<npts_ICP;i++) {
		auto bcoords = icpc[i].bcoords;
		auto v = icpc[i].v;
		int v0 = v[0] * 3, v1 = v[1] * 3, v2 = v[2] * 3;

		for(int j=0, ridx=i*3;j<nparams;j++) {
			float x = 0, y = 0, z = 0;

			x += bcoords.x * tm0RT(j, v0  ); y += bcoords.x * tm0RT(j, v0+1); z += bcoords.x * tm0RT(j, v0+2); 
			x += bcoords.y * tm0RT(j, v1  ); y += bcoords.y * tm0RT(j, v1+1); z += bcoords.y * tm0RT(j, v1+2); 
			x += bcoords.z * tm0RT(j, v2  ); y += bcoords.z * tm0RT(j, v2+1); z += bcoords.z * tm0RT(j, v2+2);

			Aexp_ICP(ridx, j) = x * w_ICP;
			Aexp_ICP(ridx+1, j) = y * w_ICP;
			Aexp_ICP(ridx+2, j) = z * w_ICP;
		}
	}

	float w_fp_scale = npts_ICP / 1000.0;
	// facial feature terms
	for(int i=0, ridx=npts_ICP*3;i<npts_feature;i++, ridx+=3) {
		auto v = targets[i].second * 3;
		float wpt = w_landmarks[i*3] * w_fp_scale;
		for(int j=0;j<nparams;j++) {
			Aexp_ICP(ridx, j) = tm0RT(j, v) * wpt;
			Aexp_ICP(ridx+1, j) = tm0RT(j, v+1) * wpt;
			Aexp_ICP(ridx+2, j) = tm0RT(j, v+2) * wpt;
		}
	}

	float w_prior_scale = npts_ICP / 1000.0;
	// prior terms
	for(int j=0;j<Aexp_ICP.cols();j++) {
		for(int i=0, ridx=npts*3;i<nparams;i++, ridx++) {
			Aexp_ICP(ridx, j) = sigma_wexp_weighted(i, j) * w_prior_scale;
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
		float wpt = w_landmarks[i*3] * w_fp_scale;

		brhs_ICP(ridx) = (q.x - Tvec.x) * wpt;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * wpt;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * wpt;
	}

	// prior term
	// fill in the lower part with the mean vector of identity weights
	int ndim_exp = mu_wexp.size();
	for(int i=0, ridx=npts*3;i<ndim_exp;i++,ridx++) {
		brhs_ICP(ridx) = mu_wexp_weighted(i) * w_prior_scale;
	}

	/*
	ofstream fout("Aexp.txt");
	PhGUtils::print2DArray(Aexp_ICP.ptr(), Aexp_ICP.cols(), Aexp_ICP.rows(), fout);
	fout.close();
	::system("pause");
	*/
	//cout << "matrix and rhs assembled." << endl;

	//cout << "least square" << endl;
	//PhGUtils::Timer tls;
	//tls.tic();
	int rtn = PhGUtils::leastsquare<double>(Aexp_ICP, brhs_ICP);
	//tls.toc("least square");
	//cout << "done." << endl;
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wexp(i) - brhs_ICP(i));
		Wexp(i) = brhs_ICP(i);		
		//cout << params[i] << ' ';
	}

	//cout << endl;

	//cout << "expression weights fitted." << endl;

	bool converged = diff / nparams < cc;
	if( converged ) cout << "expression weights converged." << endl;
	return converged;
}

void evalCost_expression_2D(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

	float s, rx, ry, rz, tx, ty, tz;
	s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

	auto targets_2d = recon->targets_2d;
	int npts = targets_2d.size();
	auto tm0cRT = recon->tm0cRT;
	auto w_landmarks = recon->w_landmarks;
	auto mu_wexp_orig = recon->mu_wexp_orig;
	auto w_prior_exp_2D = recon->w_prior_exp_2D;

	// set up rotation matrix and translation vector
	const PhGUtils::Point3f& T = recon->Tvec;
	const PhGUtils::Matrix3x3f& R = recon->Rmat;
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
    0, recon->camparams.fy, recon->camparams.cy,
    0, 0, 1);

	for(int i=0, vidx=0;i<npts;i++, vidx+=3) {
		float wpt = w_landmarks[vidx];

		float x = 0, y = 0, z = 0;
		for(int j=0;j<m;j++) {
			x += tm0cRT(j, vidx) * p[j];
			y += tm0cRT(j, vidx+1) * p[j];
			z += tm0cRT(j, vidx+2) * p[j];
		}
		const PhGUtils::Point3f& q = targets_2d[i].first;

		float u, v, d;
    PhGUtils::projectPoint(u, v, x + T.x, y + T.y, z + T.z, Mproj);

		float dx = q.x - u, dy = q.y - v, dz = q.z==0?0:(q.z - d);
		//hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
    hx[i] = sqrt(dx*dx + dy*dy);
	}

	// regularization term
  int nparams = recon->mu_wexp.n_elem;
  arma::fvec wmmu(nparams);
  for (int i = 0; i < nparams; ++i) {
    wmmu(i) = p[i] - mu_wexp_orig(i);
  }
  hx[n-1] = sqrt(arma::dot(wmmu, recon->sigma_wexp * wmmu)) * w_prior_exp_2D;
}

void evalJacobian_expression_2D(float *p, float *J, int m, int n, void* adata) {
  MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

  float s, rx, ry, rz, tx, ty, tz;
  s = p[0], rx = p[1], ry = p[2], rz = p[3], tx = p[4], ty = p[5], tz = p[6];

  auto targets_2d = recon->targets_2d;
  int npts = targets_2d.size();
  auto tm0cRT = recon->tm0cRT;
  auto w_landmarks = recon->w_landmarks;
  auto mu_wexp_orig = recon->mu_wexp_orig;
  auto w_prior_exp_2D = recon->w_prior_exp_2D;

  // set up rotation matrix and translation vector
  const PhGUtils::Point3f& T = recon->Tvec;
  const PhGUtils::Matrix3x3f& R = recon->Rmat;
  PhGUtils::Matrix3x3f Mproj(recon->camparams.fx, 0, recon->camparams.cx,
    0, recon->camparams.fy, recon->camparams.cy,
    0, 0, 1);

  float Jf[6] = { 0 };
  float f_x = recon->camparams.fx, f_y = recon->camparams.fy;

  // regularization term
  int nparams = recon->mu_wexp_orig.n_elem;
  arma::fvec wmmu(nparams);
  for (int i = 0; i < nparams; ++i) {
    wmmu(i) = p[i] - mu_wexp_orig(i);
  }
  arma::fvec Jprior = 2.0 * recon->sigma_wexp * wmmu;

  int jidx = 0;
  for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
    float wpt = w_landmarks[vidx];

    float x = 0, y = 0, z = 0;
    for (int j = 0; j < m; j++) {
      x += tm0cRT(j, vidx) * p[j];
      y += tm0cRT(j, vidx + 1) * p[j];
      z += tm0cRT(j, vidx + 2) * p[j];
    }
    const PhGUtils::Point3f& q = targets_2d[i].first;

    float Px = x + T.x, Py = y + T.y, Pz = z + T.z;
    float u, v;
    PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

    float dx = u - q.x, dy = v - q.y;

    float rk = sqrt(dx*dx + dy*dy);
    float inv_rk = 1.0 / rk;

    // Jf
    float inv_z = 1.0 / Pz;
    float inv_z2 = inv_z * inv_z;

    Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
    Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

    for (int j = 0; j < m; ++j) {
      // Jm * e_j
      float jmex = tm0cRT(j, vidx), jmey = tm0cRT(j, vidx + 1), jmez = tm0cRT(j, vidx + 2);
      // Jf * Jm * e_j
      float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
      float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
      // data term and prior term
      J[jidx++] = (jfjmx * dx + jfjmy * dy) * inv_rk;
    }
  }

  for (int j = 0; j < m; ++j) {
    J[jidx++] = Jprior(j) * w_prior_exp_2D;
  }
}

bool MultilinearReconstructor_old::fitExpressionWeights_withPrior_2D()
{
	// fix both rotation and identity weights, solve for expression weights
	int nparams = core.dim(1);
	vector<float> params(Wexp.rawptr(), Wexp.rawptr() + nparams);
	int npts = targets.size();
	vector<float> meas(npts+1, 0.0);
#if EXPRESSION_FIT_USE_DIFF
  float opts[5] = { 1e-3, 1e-12, 1e-12, 1e-12, 1e-6 };
  int iters = slevmar_dif(evalCost_expression_2D, &(params[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, this);
#else
  float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9};
  int iters = slevmar_der(evalCost_expression_2D, evalJacobian_expression_2D, &(params[0]), &(meas[0]), nparams, npts + 1, 128, NULL, NULL, NULL, NULL, this);
#endif

	//cout << "finished in " << iters << " iterations." << endl;

	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff = max(diff, fabs(Wexp(i) - params[i]));
		Wexp(i) = params[i];
	}

	return diff < 1e-6;
}

void evalCost3(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);

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

bool MultilinearReconstructor_old::fitExpressionWeights_withPrior()
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
	int rtn = PhGUtils::leastsquare<float>(Aexp, brhs);
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
void MultilinearReconstructor_old::transformMesh()
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
	#pragma omp parallel for
	for(int i=0;i<nverts;i++) {
		int idx = i * 3;
		tmesh(idx) = pt_trans(0, i) + T(0);
		tmesh(idx+1) = pt_trans(1, i) + T(1);
		tmesh(idx+2) = pt_trans(2, i) + T(2);
	}
}

void MultilinearReconstructor_old::bindRGBDTarget(const vector<unsigned char>& colordata, const vector<unsigned char>& depthdata)
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

void MultilinearReconstructor_old::bindTarget( const vector<pair<PhGUtils::Point3f, int>>& pts, TargetType ttp )
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
		float w_depth = exp(-dz * dz / (sigma_depth*sigma_depth*10000.0));

		// set the landmark weights
#if 0
		w_landmarks[idx] = w_landmarks[idx+1] = w_landmarks[idx+2] = (i<64 || i>74)?isValid*w_depth:isValid*w_boundary*w_depth;
#else
    w_landmarks[idx] = w_landmarks[idx + 1] = w_landmarks[idx + 2] = 1.0;
#endif

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
#if 0
    /// for 78 points model
		RTparams[6] = fabs((targets[16].first.x + targets[20].first.x) * 0.5
                      -(targets[24].first.x + targets[28].first.x) * 0.5);
#else
    /// for 68 points model
    RTparams[6] = fabs((targets[36].first.x + targets[39].first.x) * 0.5
      - (targets[42].first.x + targets[45].first.x) * 0.5); 
#endif
		// set the initial mean RT
    cout << "initial rigid transformation params: ";
    for (int i = 0; i < 7; i++){
      meanRT[i] = RTparams[i];
      cout << meanRT[i] << "\t";
    }
    cout << endl;
	}

	//PhGUtils::debug("valid landmarks", validCount);
}

void MultilinearReconstructor_old::updateMatrices() {
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

void MultilinearReconstructor_old::updateComputationTensor()
{
	updateCoreC();
	tm0c = corec.modeProduct(Wid, 0);
	tm0cRT = tm0c;
	tm1c = corec.modeProduct(Wexp, 1);
	tm1cRT = tm1c;
	tmc = tm1c.modeProduct(Wid, 0);
}

// build a truncated version of the core
void MultilinearReconstructor_old::updateCoreC() {
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
void MultilinearReconstructor_old::transformTM0()
{
	int npts = tm0.dim(1) / 3;
	// don't use the assignment operator, it actually moves the data
	//tm0cRT = tm0c;
	#pragma omp parallel for
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
void MultilinearReconstructor_old::transformTM1()
{
	//cout << "Transform TM1 ..." << endl;
	int npts = tm1.dim(1) / 3;
	// don't use the assignment operator, it actually moves the data
	//tm1cRT = tm1c;
	#pragma omp parallel for
	for(int i=0;i<tm1.dim(0);i++) {
		for(int j=0, vidx=0;j<npts;j++, vidx+=3) {

			tm1RT(i, vidx) = tm1(i, vidx);
			tm1RT(i, vidx+1) = tm1(i, vidx+1);
			tm1RT(i, vidx+2) = tm1(i, vidx+2);

			// rotation only!!!
			PhGUtils::rotatePoint( tm1RT(i, vidx), tm1RT(i, vidx+1), tm1RT(i, vidx+2), Rmat );
		}
	}
	//cout << "done." << endl;
}

// transform TM0C with global rigid transformation
void MultilinearReconstructor_old::transformTM0C() {
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
void MultilinearReconstructor_old::transformTM1C() {
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

void MultilinearReconstructor_old::updateTMC() {
	tm1c.modeProduct(Wid, 0, tmc);
}

void MultilinearReconstructor_old::updateTMCwithTM0C() {
	//PhGUtils::message("updating tmc");
	tm0c.modeProduct(Wexp, 0, tmc);
}

void MultilinearReconstructor_old::updateTM()
{
	tm1.modeProduct(Wid, 0, tplt);
}

void MultilinearReconstructor_old::updateTMwithTM0()
{
	tm0.modeProduct(Wexp, 0, tplt);
}

float MultilinearReconstructor_old::computeError_ICP() {

	// set up rotation matrix and translation vector
	float w_fp_scale = icpc.size() / 1000.0;
	float E = 0, wsum = 0;
	// ICP terms
	for(int i=0;i<icpc.size();i++) {
		auto v = icpc[i].v;
		auto bcoords = icpc[i].bcoords;

		int vidx[3];
		vidx[0] = v.x*3; vidx[1] = v.y*3; vidx[2] = v.z*3;

		PhGUtils::Point3f p;
		p.x += bcoords.x * tplt(vidx[0]); p.y += bcoords.x * tplt(vidx[0]+1); p.z += bcoords.x * tplt(vidx[0]+2);
		p.x += bcoords.y * tplt(vidx[1]); p.y += bcoords.y * tplt(vidx[1]+1); p.z += bcoords.y * tplt(vidx[1]+2);
		p.x += bcoords.z * tplt(vidx[2]); p.y += bcoords.z * tplt(vidx[2]+1); p.z += bcoords.z * tplt(vidx[2]+2);

		const PhGUtils::Point3f& q = icpc[i].q;

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, Rmat, Tvec );

		float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
		E += (dx * dx + dy * dy + dz * dz) * w_ICP;
		wsum += w_ICP;
	}

	// facial feature term
	for(int i=0;i<targets.size();i++) {
		int vidx = targets[i].second * 3;
		float wpt = w_landmarks[i*3] * w_fp_scale;

		PhGUtils::Point3f p(tplt(vidx), tplt(vidx+1), tplt(vidx+2));

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, Rmat, Tvec );

		// for mouth region and outer contour, use only 2D info
		if( i < 42 || i > 74 ) {
			const PhGUtils::Point3f& q = targets[i].first;

			float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
			E += (dx * dx + dy * dy + dz * dz) * wpt;
			wsum += wpt;
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
			const PhGUtils::Point3f& q = targets_2d[i].first;

			//cout << "#" << i <<"\t" << u << ", " << v << "\t" << q.x << ", " << q.y << endl;

			float du = u - q.x, dv = v - q.y;
			E += (du * du + dv * dv) * wpt;
			wsum += wpt;
		}
	}
	E /= wsum;
	return E;
}

float MultilinearReconstructor_old::computeError()
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

float MultilinearReconstructor_old::computeError_2D()
{
	int npts = targets.size();
	float E = 0;
  PhGUtils::Matrix3x3f Mproj(camparams.fx, 0, camparams.cx,
    0, camparams.fy, camparams.cy,
    0, 0, 1);
	for(int i=0;i<npts;i++) {
		int vidx = i * 3;
		// should change this to a fast version
		PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));

		/*p = Rmat * p + Tvec;*/		
		PhGUtils::transformPoint(p.x, p.y, p.z, Rmat, Tvec);

		// project the point
		float u, v;
    PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Mproj);

		float dx, dy;
		dx = u - targets_2d[i].first.x;
		dy = v - targets_2d[i].first.y;
		E += (dx * dx + dy * dy) * w_landmarks[vidx];
	}


	E /= npts;
	return E;
}

// render the mesh to the FBO
// need to call update mesh before rendering it, if the latest mesh is needed
void MultilinearReconstructor_old::renderMesh()
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

void MultilinearReconstructor_old::updateMesh()
{
	#pragma omp parallel for
	for(int i=0;i<tplt.length()/3;i++) {
		int idx = i * 3;
		baseMesh.vertex(i).x = tmesh(idx++);
		baseMesh.vertex(i).y = tmesh(idx++);
		baseMesh.vertex(i).z = tmesh(idx);
	}

#if 0
	PhGUtils::OBJWriter writer;
	writer.save(baseMesh, "../Data/tmesh.obj");
#endif
}

// mesh fitting functions

tuple<vector<float>, vector<float>, vector<float>> MultilinearReconstructor_old::fitMesh(const string& filename, const vector<pair<int, int>>& hint)
{
	// reset all related parameters
	reset();
	cout << "fitting mesh with file " << filename << endl;
	// load the target mesh
	shared_ptr<PhGUtils::TriMesh> msh = shared_ptr<PhGUtils::TriMesh>(new PhGUtils::TriMesh);
	PhGUtils::OBJLoader loader;
	loader.load(filename);
	msh->initWithLoader(loader);

	PhGUtils::MeshViewer* viewer = new PhGUtils::MeshViewer;
	viewer->bindMesh(msh);
	viewer->bindHints(hint);
	viewer->show();

	// convert hint to vertex-to-point constraint
	vcons.resize(hint.size());
	for(int i=0;i<vcons.size();i++) {
		vcons[i].q = msh->vertex(hint[i].first);
		vcons[i].vidx = hint[i].second;
	}

	// pre-registration for rigid transformation ONLY with the given correspondence
	fit_withConstraints();
	
	idxvec.resize(msh->vertCount());
	for(int i=0;i<idxvec.size();i++) idxvec[i] = i;

	// ICP registration
	fitMesh_ICP(msh);

	// for debug, write out the fitted mesh
	PhGUtils::OBJWriter writer;
	writer.save(baseMesh, "../Data/tmesh.obj");

	// the the fitted parameters and return them as a tuple
	vector<float> vRT(RTparams, RTparams+7), vWid = Wid.toStdVector(), vWexp = Wexp.toStdVector();	
	return make_tuple(vRT, vWid, vWexp);
}

void evalCost_withConstraints(float *p, float *hx, int m, int n, void* adata) {
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);
	auto targets = recon->vcons;
	auto tplt = recon->tplt;

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];
	PhGUtils::Point3f T(tx, ty, tz);
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz) * s;

	for(int i=0;i<n;i++) {
		int voffset = targets[i].vidx * 3;
		const PhGUtils::Point3f& q = targets[i].q;
		PhGUtils::Point3f p(tplt(voffset),tplt(voffset+1), tplt(voffset+2));

		PhGUtils::transformPoint( p.x, p.y, p.z, R, T );

		hx[i] = p.squaredDistanceTo(q);
	}
}

void evalJacobian_withConstraints(float *p, float *J, int m, int n, void *adata) {
//PhGUtils::message("jac func");
	// J is a n-by-m matrix
	MultilinearReconstructor_old* recon = static_cast<MultilinearReconstructor_old*>(adata);
	auto targets = recon->vcons;
	auto tplt = recon->tplt;

	float s, rx, ry, rz, tx, ty, tz;
	rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5], s = p[6];

	// set up rotation matrix and translation vector
	PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
	PhGUtils::Matrix3x3f Jx, Jy, Jz;
	PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);

	// apply the new global transformation
	for(int i=0, vidx=0, jidx=0;i<n;i++) {
		// point p
		int voffset = targets[i].vidx * 3;
		const PhGUtils::Point3f& q = targets[i].q;
		PhGUtils::Point3f p(tplt(voffset),tplt(voffset+1), tplt(voffset+2));

		// R * p
		float rpx = p.x, rpy = p.y, rpz = p.z;
		PhGUtils::rotatePoint( rpx, rpy, rpz, R );

		// s * R * p + t - q
		float rkx = s * rpx + tx - q.x, rky = s * rpy + ty - q.y, rkz = s * rpz + tz - q.z;

		float jpx, jpy, jpz;
		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jx );
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jy );
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		jpx = p.x, jpy = p.y, jpz = p.z;
		PhGUtils::rotatePoint( jpx, jpy, jpz, Jz );
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * (jpx * rkx + jpy * rky + jpz * rkz);

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * rkx;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * rky;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * rkz;

		// \frac{\partial r_i}{\partial s}
		J[jidx++] = 2.0 * (rpx * rkx + rpy * rky + rpz * rkz);
	}
}

void MultilinearReconstructor_old::fitIdentityWeights_withPrior_Constraints() {
	//cout << "fitting identity weights ..." << endl;
	// to use this method, the tensor tm1 must first be updated using the rotation matrix
	int nparams = core.dim(0);	
	int npts = vcons.size();

	Aid_ICP = PhGUtils::DenseMatrix<double>(npts*3 + nparams, nparams);
	brhs_ICP = PhGUtils::DenseVector<double>(npts*3 + nparams);

	float wpt_scale = 1e-3;
	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	int ridx = 0;
	for(int i=0;i<npts;i++, ridx+=3) {
		int voffset = vcons[i].vidx * 3;
		float wpt = w_ICP * wpt_scale;

		for(int j=0;j<nparams;j++) {
			Aid_ICP(ridx  , j) = tm1RT(j, voffset  ) * wpt;
			Aid_ICP(ridx+1, j) = tm1RT(j, voffset+1) * wpt;
			Aid_ICP(ridx+2, j) = tm1RT(j, voffset+2) * wpt;
		}
	}

	float w_prior_scale = 1.0;
	// prior terms
	for(int j=0;j<Aid_ICP.cols();j++) {
		for(int i=0, ridx=npts*3;i<nparams;i++, ridx++) {
			Aid_ICP(ridx, j) = sigma_wid_weighted(i, j) * w_prior_scale;
		}
	}

	//PhGUtils::Matrix3x3f invRmat = Rmat.inv();


	// assemble the right hand side, fill in the upper part as usual
	// ICP terms
	for(int i=0, ridx=0;i<npts;ridx+=3, i++) {
		const PhGUtils::Point3f& q = vcons[i].q;
		float wpt = w_ICP * wpt_scale;

		brhs_ICP(ridx) = (q.x - Tvec.x) * wpt;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * wpt;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * wpt;
	}

	// prior term
	// fill in the lower part with the mean vector of identity weights
	int ndim_id = mu_wid.size();
	for(int i=0, ridx=npts*3;i<ndim_id;i++,ridx++) {
		brhs_ICP(ridx) = mu_wid_weighted(i) * w_prior_scale;
	}

	//cout << "matrix and rhs assembled." << endl;

	//cout << "least square" << endl;
#if USE_MKL_LS
	int rtn = PhGUtils::leastsquare<double>(Aid_ICP, brhs_ICP);
#else
	int rtn = leastsquare_normalmat(Aid_ICP, brhs_ICP, AidtAid, Aidtb);
#endif
	//cout << "done." << endl;
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
#if USE_MKL_LS
		diff += fabs(Wid(i) - brhs_ICP(i));
		Wid(i) = brhs_ICP(i);		
#else
		diff += fabs(Wid(i) - Aidtb(i));
		Wid(i) = Aidtb(i);	
#endif
		//cout << params[i] << ' ';
	}

	//cout << endl;

	//cout << "identity weights fitted." << endl;
}

void MultilinearReconstructor_old::fitExpressionWeights_withPrior_Constraints()
{
	//cout << "fitting expression weights ..." << endl;
	// to use this method, the tensor tm1 must first be updated using the rotation matrix
	int nparams = core.dim(1);
	int npts = vcons.size();

	Aexp_ICP = PhGUtils::DenseMatrix<double>(npts*3 + nparams, nparams);
	brhs_ICP = PhGUtils::DenseVector<double>(npts*3 + nparams);

	// assemble the matrix, fill in the upper part
	// the lower part is already filled in
	// ICP terms
	float wpt_scale = 1e-3;
	int ridx = 0;
	for(int i=0;i<npts;i++, ridx+=3) {
		int voffset = vcons[i].vidx * 3;
		float wpt = w_ICP * wpt_scale;

		for(int j=0;j<nparams;j++) {

			Aexp_ICP(ridx  , j) = tm0RT(j, voffset  ) * wpt;
			Aexp_ICP(ridx+1, j) = tm0RT(j, voffset+1) * wpt;
			Aexp_ICP(ridx+2, j) = tm0RT(j, voffset+2) * wpt;
		}
	}

	float w_prior_scale = 1.0;
	// prior terms
	for(int j=0;j<Aexp_ICP.cols();j++) {
		for(int i=0, ridx=npts*3;i<nparams;i++, ridx++) {
			Aexp_ICP(ridx, j) = sigma_wexp_weighted(i, j) * w_prior_scale;
		}
	}

	//PhGUtils::Matrix3x3f invRmat = Rmat.inv();


	// assemble the right hand side, fill in the upper part as usual
	// ICP terms
	for(int i=0, ridx=0;i<npts;ridx+=3, i++) {
		const PhGUtils::Point3f& q = vcons[i].q;
		float wpt = w_ICP * wpt_scale;

		brhs_ICP(ridx) = (q.x - Tvec.x) * wpt;
		brhs_ICP(ridx+1) = (q.y - Tvec.y) * wpt;
		brhs_ICP(ridx+2) = (q.z - Tvec.z) * wpt;
	}

	// prior term
	// fill in the lower part with the mean vector of identity weights
	int ndim_exp = mu_wexp.size();
	for(int i=0, ridx=npts*3;i<ndim_exp;i++,ridx++) {
		brhs_ICP(ridx) = mu_wexp_weighted(i) * w_prior_scale;
	}

	//cout << "matrix and rhs assembled." << endl;

	//cout << "least square" << endl;
	int rtn = PhGUtils::leastsquare<double>(Aexp_ICP, brhs_ICP);
	//cout << "done." << endl;
	//debug("rtn", rtn);
	float diff = 0;
	//b.print("b");
	for(int i=0;i<nparams;i++) {
		diff += fabs(Wexp(i) - brhs_ICP(i));
		Wexp(i) = brhs_ICP(i);		
		//cout << params[i] << ' ';
	}

	//cout << endl;

	//cout << "expression weights fitted." << endl;
}

void MultilinearReconstructor_old::fit_withConstraints()
{
	PhGUtils::message("fit with point constraints...");
	int npts = vcons.size();
	RTparams[6] = 100.0;
	vector<float> meas(npts);
	

	int iters = 0;
	float E0 = 0;
	bool converged = false;
	const int MaxIterations = 32;
	while( !converged && iters++ < MaxIterations ) {
		// fit rigid transformation
		//int iters = slevmar_dif(evalCost_withConstraints, &(RTparams[0]), &(meas[0]), 7, npts, 1024, NULL, NULL, NULL, NULL, this);
		int iters = slevmar_der(evalCost_withConstraints, evalJacobian_withConstraints, &(RTparams[0]), &(meas[0]), 7, npts, 1024, NULL, NULL, NULL, NULL, this);
		/*
		PhGUtils::message("rigid transformation estimated in " + PhGUtils::toString(iters) + " iterations.");
		PhGUtils::printArray(RTparams, 7);
		*/

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

		
		// fit identity
		transformTM1();
		fitIdentityWeights_withPrior_Constraints();

		tm0 = core.modeProduct(Wid, 0);
		
		// fit expression
		transformTM0();		
		fitExpressionWeights_withPrior_Constraints();

		tm1 = core.modeProduct(Wexp, 1);
		
		updateTM();		

		E = computeError_Constraints();
		PhGUtils::debug("iters", iters, "Error", E);

		converged |= E < errorThreshold;
		converged |= fabs(E - E0) < errorDiffThreshold;

		E0 = E;
	}
}

void MultilinearReconstructor_old::collectICPConstraints(const shared_ptr<PhGUtils::TriMesh>& msh, int iter, int maxIter)
{
	const float scaleFactor = fabs(RTparams[6]);
	// determine a distance threshold
	const float DIST_THRES_MAX = 0.1 * scaleFactor;
	const float DIST_THRES_MIN = 0.01 * scaleFactor;
	float DIST_THRES = DIST_THRES_MAX + (DIST_THRES_MIN - DIST_THRES_MAX) * iter / (float)maxIter;
	PhGUtils::message("Collecting ICP constraints...");
	icpc.clear();
	icpc.reserve(32768);

	baseMesh.updateAABB();

	int nverts = tplt.length()/3;

	srand(time(NULL));

	random_shuffle(idxvec.begin(), idxvec.end());

	int maxSamples = idxvec.size() / 20;
	int minSamples = idxvec.size() / 200;

	int nsamples = iter/(float)maxIter * (maxSamples-minSamples)+minSamples;

	// for each vertex on the template mesh, find the closest point on the target mesh
	#pragma omp parallel for
	for(int i=0;i<nsamples;i++) {
		ICPConstraint cc;
		cc.q = msh->vertex(idxvec[i]);		
		PhGUtils::Point3f bcoords;

#define BRUTEFOREC_ICP 0
#if BRUTEFORCE_ICP
		float dist = baseMesh.findClosestPoint_bruteforce(cc.q, cc.v, cc.bcoords);
		if( dist < DIST_THRES ) {
#else
		float dist = baseMesh.findClosestPoint(cc.q, cc.v, cc.bcoords, DIST_THRES);
		if( dist > 0 ) {
#endif
			// add a constraint
			#pragma omp critical
			{
			icpc.push_back(cc);
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

void MultilinearReconstructor_old::fitMesh_ICP(
	const shared_ptr<PhGUtils::TriMesh>& msh)
{
	fitPose = true; fitIdentity = true; fitExpression = true;
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
	const int MaxIterations = 32;
	while( !converged && iters++ < MaxIterations ) {
		converged = true;

		// update mesh and render the mesh
		transformMesh();
		updateMesh();

		// collect ICP constraints
		collectICPConstraints(msh, iters, MaxIterations);

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
		E = computeError_ICP();
		PhGUtils::debug("iters", iters, "Error", E);

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (icpc.size()/5000.0));
		converged |= fabs(E - E0) < errorDiffThreshold_ICP;
		E0 = E;
		timerOther.toc();
		//emit oneiter();
		//QApplication::processEvents();
	}

	timerTransform.tic();
	
	if( fitIdentity && fitExpression ) {
		tplt = core.modeProduct(Wexp, 1).modeProduct(Wid, 0);
	}
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
	timerTotal.toc("total time");
}

float MultilinearReconstructor_old::computeError_Constraints()
{
	// set up rotation matrix and translation vector
	float E = 0, wsum = 0;
	// ICP terms
	for(int i=0;i<vcons.size();i++) {
		int voffset = vcons[i].vidx * 3;

		PhGUtils::Point3f p(tplt(voffset), tplt(voffset+1), tplt(voffset+2));

		const PhGUtils::Point3f& q = vcons[i].q;

		// p = R * p + T
		PhGUtils::transformPoint( p.x, p.y, p.z, Rmat, Tvec );

		float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
		E += (dx * dx + dy * dy + dz * dz);
	}

	return E / vcons.size();
}

void MultilinearReconstructor_old::printStats()
{

}
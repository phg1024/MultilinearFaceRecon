#include "MultilinearReconstructorGPU.cuh"
#include <helper_math.h>
#include <helper_functions.h>
#include "Kinect/KinectUtils.h"
#include "Utils/Timer.h"

#include "Elements_GPU.h"

#define FBO_DEBUG_GPU 0
#define KERNEL_DEBUG 0
#define OUTPUT_ICPC 0

MultilinearReconstructorGPU::MultilinearReconstructorGPU():
	d_tu0(nullptr), d_tu1(nullptr), d_tm0(nullptr), d_tm1(nullptr),
	d_tplt(nullptr), d_mesh(nullptr), d_tm0RT(nullptr), d_tm1RT(nullptr),
	d_fptsIdx(nullptr), d_q2d(nullptr), d_q(nullptr), 
	d_colordata(nullptr), d_depthdata(nullptr),
	d_targetLocations(nullptr), d_RTparams(nullptr),
	d_A(nullptr), d_b(nullptr), d_meshtopo(nullptr)
{
	// set device
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	checkCudaState();

	cudaSetDeviceFlags(cudaDeviceMapHost);

	PhGUtils::message("initializing CULA ...");
	culaInitialize();
	checkCudaState();
	PhGUtils::message("creating CUDA stream ...");
	cudaStreamCreate(&mystream);
	checkCudaState();

	w_prior_id = 1e-3;
	w_prior_exp = 1e-4;
	w_boundary = 1e-6;

	meanX = meanY = meanZ = 0;

	// initialize offscreen renderer
	initRenderer();
	
	// initialize members
	init();

	// process the loaded data
	preprocess();

	cudaDeviceSynchronize();
}

MultilinearReconstructorGPU::~MultilinearReconstructorGPU() {
	// release resources
	cudaDeviceReset();
}

__host__ void MultilinearReconstructorGPU::setPose(const float* params) {
	for(int i=0;i<7;i++) h_RTparams[i] = params[i];
	checkCudaErrors(cudaMemcpy(d_RTparams, params, sizeof(float)*7, cudaMemcpyHostToDevice));
}

__host__ void MultilinearReconstructorGPU::setIdentityWeights(const Tensor1<float>& t) {
	// copy to GPU
	checkCudaErrors(cudaMemcpy(d_Wid, t.rawptr(), sizeof(float)*ndims_wid, cudaMemcpyHostToDevice));
}

__host__ void MultilinearReconstructorGPU::setExpressionWeights(const Tensor1<float>& t) {
	// copy to GPU
	checkCudaErrors(cudaMemcpy(d_Wexp, t.rawptr(), sizeof(float)*ndims_wexp, cudaMemcpyHostToDevice));
}

__host__ void MultilinearReconstructorGPU::preprocess() {
	PhGUtils::message("preprocessing the input data...");

	// process the identity prior

	// invert sigma_wid
	int* ipiv;
	checkCudaErrors(cudaMalloc((void**) &ipiv, sizeof(int)*ndims_wid));
	culaDeviceSgetrf(ndims_wid, ndims_wid, d_sigma_wid, ndims_wid, ipiv);
	culaDeviceSgetri(ndims_wid, d_sigma_wid, ndims_wid, ipiv);
	checkCudaErrors(cudaFree(ipiv));

	// multiply inv_sigma_wid to mu_wid
	cublasSgemv('N', ndims_wid, ndims_wid, 1.0, d_sigma_wid, ndims_wid, d_mu_wid, 1, 0.0, d_mu_wid_weighted, 1); 
	
	// scale inv_sigma_wid by w_prior_id
	cublasSscal(ndims_wid*ndims_wid, w_prior_id, d_sigma_wid, 1);

	// scale mu_wid by w_prior_id
	cublasSscal(ndims_wid, w_prior_id, d_mu_wid_weighted, 1);

	// copy back the inverted matrix to check correctness
	writeback(d_sigma_wid, ndims_wid*ndims_wid, "invswid.txt");

	// process the expression prior

	// invert sigma_wexp
	checkCudaErrors(cudaMalloc((void**) &ipiv, sizeof(int)*ndims_wexp));
	culaDeviceSgetrf(ndims_wexp, ndims_wexp, d_sigma_wexp, ndims_wexp, ipiv);
	culaDeviceSgetri(ndims_wexp, d_sigma_wexp, ndims_wexp, ipiv);
	checkCudaErrors(cudaFree(ipiv));

	// multiply inv_sigma_wexp to mu_wexp
	cublasSgemv('N', ndims_wexp, ndims_wexp, 1.0, d_sigma_wexp, ndims_wexp, d_mu_wexp, 1, 0, d_mu_wexp_weighted, 1);

	// scale inv_sigma_wexp by w_prior_exp
	cublasSscal(ndims_wexp*ndims_wexp, w_prior_exp, d_sigma_wexp, 1);

	// scale mu_wexp by w_prior_exp
	cublasSscal(ndims_wexp, w_prior_exp, d_mu_wexp_weighted, 1); 

	writeback(d_sigma_wexp, ndims_wexp*ndims_wexp, "invswexp.txt");
	PhGUtils::message("done.");

	// initialize Wid and Wexp
	checkCudaErrors(cudaMemcpy(d_Wid, d_mu_wid0, sizeof(float)*ndims_wid, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_Wexp, d_mu_wexp0, sizeof(float)*ndims_wexp, cudaMemcpyDeviceToDevice));

	// initialize tm0, tm1

	// tm0 = tu0 * Wid, use cublas
	// tu0: ndims_wid * (ndims_wexp * ndims_pts) matrix, each row corresponds to an identity
	//		inside each row, the vertices are arranged by expression
	//		That is, a row in tu0 can be see as a row-major matrix where each row corresponds to an expression
	// tm0: a row-major matrix where each row corresponds to an expression
	cublasSgemv('N', ndims_wexp * ndims_pts, ndims_wid, 1.0, d_tu0, ndims_wexp * ndims_pts, d_Wid, 1, 0, d_tm0, 1);
	writeback(d_tm0, ndims_wexp, ndims_pts, "tm0.txt");

	// tm1 = tu1 * Wexp, use cublas
	// tu1: ndims_wexp * (ndims_wid * ndims_pts) matrix, each row corresponds to an expression
	//		inside each row, the vertices are arraged using index-major
	//		That is, a row in tu1 can be see as a column-major matrix where each column corresponds to an identity
	// tm1: a column-major matrix where each column corresponds to an identity
	cublasSgemv('N', ndims_wid * ndims_pts, ndims_wexp, 1.0, d_tu1, ndims_wid * ndims_pts, d_Wexp, 1, 0, d_tm1, 1);
	writeback(d_tm1, ndims_pts, ndims_wid, "tm1.txt");

	// create template mesh
	// tplt = tm1 * Wid, use cublas
	cublasSgemv('T', ndims_wid, ndims_pts, 1.0, d_tm1, ndims_wid, d_Wid, 1, 0.0, d_tplt, 1);
	writeback(d_tplt, ndims_pts/3, 3, "tplt.txt");
}

__host__ void MultilinearReconstructorGPU::init() {
	showCUDAMemoryUsage();
	// read the core tensor
	PhGUtils::message("Loading core tensor ...");
	const string filename = "../Data/blendshape/core.bin";

	Tensor3<float> core;
	core.read(filename);
	core_dim[0] = core.dim(0), core_dim[1] = core.dim(1), core_dim[2] = core.dim(2);
	int totalSize = core_dim[0] * core_dim[1] * core_dim[2];

	tmesh.resize(core_dim[2]);

	// unfold it
	Tensor2<float> tu0 = core.unfold(0), tu1 = core.unfold(1);

	PhGUtils::message("transferring the unfolded core tensor to GPU ...");

#if 1
	checkCudaErrors(cudaHostAlloc((void**) &h_tu0, sizeof(float)*totalSize, cudaHostAllocMapped));
	memcpy(h_tu0, tu0.rawptr(), sizeof(float)*totalSize);
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_tu0, h_tu0, 0));

	checkCudaErrors(cudaHostAlloc((void**) &h_tu1, sizeof(float)*totalSize, cudaHostAllocMapped));
	memcpy(h_tu1, tu1.rawptr(), sizeof(float)*totalSize);
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_tu1, h_tu1, 0));
#else
	// transfer the unfolded core tensor to GPU
	checkCudaErrors(cudaMalloc((void **) &d_tu0, sizeof(float)*totalSize));
	checkCudaErrors(cudaMemcpy(d_tu0, tu0.rawptr(), sizeof(float)*totalSize, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_tu1, sizeof(float)*totalSize));
	checkCudaErrors(cudaMemcpy(d_tu1, tu1.rawptr(), sizeof(float)*totalSize, cudaMemcpyHostToDevice));
#endif

	PhGUtils::message("done.");
	showCUDAMemoryUsage();

	PhGUtils::message("allocating memory for computation (tensors) ...");
	// allocate memory for the tm0, tm1, tm0RT, tm1RT, tplt
	checkCudaErrors(cudaMalloc((void **) &d_tm0, sizeof(float)*core_dim[1]*core_dim[2]));
	checkCudaErrors(cudaMalloc((void **) &d_tm0RT, sizeof(float)*core_dim[1]*core_dim[2]));
	checkCudaErrors(cudaMalloc((void **) &d_tm1, sizeof(float)*core_dim[0]*core_dim[2]));
	checkCudaErrors(cudaMalloc((void **) &d_tm1RT, sizeof(float)*core_dim[0]*core_dim[2]));
	checkCudaErrors(cudaMalloc((void **) &d_tplt, sizeof(float)*core_dim[2]));
	checkCudaErrors(cudaMalloc((void **) &d_mesh, sizeof(float)*core_dim[2]));
	checkCudaErrors(cudaMemset(d_mesh, 0, sizeof(float)*core_dim[2]));
	showCUDAMemoryUsage();

	// read the prior
	PhGUtils::message("Loading prior data ...");

	// identity prior
	PhGUtils::message("Loading identity prior data ...");
	const string fnwid  = "../Data/blendshape/wid.bin";

	ifstream fwid(fnwid, ios::in | ios::binary );
	fwid.read(reinterpret_cast<char*>(&ndims_wid), sizeof(int));
	cout << "identity prior dim = " << ndims_wid << endl;
	vector<float> mu_wid0, mu_wid, sigma_wid;
	mu_wid0.resize(ndims_wid);
	mu_wid.resize(ndims_wid);
	sigma_wid.resize(ndims_wid*ndims_wid);

	fwid.read(reinterpret_cast<char*>(&(mu_wid0[0])), sizeof(float)*ndims_wid);
	fwid.read(reinterpret_cast<char*>(&(mu_wid[0])), sizeof(float)*ndims_wid);
	fwid.read(reinterpret_cast<char*>(&(sigma_wid[0])), sizeof(float)*ndims_wid*ndims_wid);

	fwid.close();

	PhGUtils::message("identity prior loaded.");
	PhGUtils::message("transferring identity prior to GPU ...");

	// transfer the identity prior to GPU
	checkCudaErrors(cudaMalloc((void **) &d_mu_wid0, sizeof(float)*ndims_wid));
	checkCudaErrors(cudaMemcpy(d_mu_wid0, &(mu_wid0[0]), sizeof(float)*ndims_wid, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_mu_wid, sizeof(float)*ndims_wid));
	checkCudaErrors(cudaMemcpy(d_mu_wid, &(mu_wid[0]), sizeof(float)*ndims_wid, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**) &d_mu_wid_weighted, sizeof(float)*ndims_wid));
	checkCudaErrors(cudaMalloc((void**) &d_Wid, sizeof(float)*ndims_wid));

	checkCudaErrors(cudaMalloc((void **) &d_sigma_wid, sizeof(float)*ndims_wid*ndims_wid));
	checkCudaErrors(cudaMemcpy(d_sigma_wid, &(sigma_wid[0]), sizeof(float)*ndims_wid*ndims_wid, cudaMemcpyHostToDevice));

	// write back for examiniation
	PhGUtils::write2file(sigma_wid, "wid.txt");

	PhGUtils::message("done.");
	showCUDAMemoryUsage();

	// expression prior
	PhGUtils::message("Loading expression prior data ...");
	const string fnwexp = "../Data/blendshape/wexp.bin";
	ifstream fwexp(fnwexp, ios::in | ios::binary );

	fwexp.read(reinterpret_cast<char*>(&ndims_wexp), sizeof(int));
	cout << "expression prior dim = " << ndims_wexp << endl;
	vector<float> mu_wexp0, mu_wexp, sigma_wexp;
	mu_wexp0.resize(ndims_wexp);
	mu_wexp.resize(ndims_wexp);
	sigma_wexp.resize(ndims_wexp*ndims_wexp);

	fwexp.read(reinterpret_cast<char*>(&(mu_wexp0[0])), sizeof(float)*ndims_wexp);
	fwexp.read(reinterpret_cast<char*>(&(mu_wexp[0])), sizeof(float)*ndims_wexp);
	fwexp.read(reinterpret_cast<char*>(&(sigma_wexp[0])), sizeof(float)*ndims_wexp*ndims_wexp);

	fwexp.close();

	PhGUtils::message("expression prior loaded.");
	PhGUtils::message("transferring expression prior to GPU ...");

	// transfer the expression prior to GPU
	checkCudaErrors(cudaMalloc((void **) &d_mu_wexp0, sizeof(float)*ndims_wexp));
	checkCudaErrors(cudaMemcpy(d_mu_wexp0, &(mu_wexp0[0]), sizeof(float)*ndims_wexp, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_mu_wexp, sizeof(float)*ndims_wexp));
	checkCudaErrors(cudaMemcpy(d_mu_wexp, &(mu_wexp[0]), sizeof(float)*ndims_wexp, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**) &d_mu_wexp_weighted, sizeof(float)*ndims_wexp));
	checkCudaErrors(cudaMalloc((void**) &d_Wexp, sizeof(float)*ndims_wexp));

	checkCudaErrors(cudaMalloc((void **) &d_sigma_wexp, sizeof(float)*ndims_wexp*ndims_wexp));
	checkCudaErrors(cudaMemcpy(d_sigma_wexp, &(sigma_wexp[0]), sizeof(float)*ndims_wexp*ndims_wexp, cudaMemcpyHostToDevice));

	// write back for examination
	PhGUtils::write2file(sigma_wexp, "wexp.txt");

	PhGUtils::message("done.");
	showCUDAMemoryUsage();

	// load the indices of landmarks
	const string lmfn = "../Data/model/landmarks.txt";
	ifstream fin(lmfn, ios::in);
	if( fin.is_open() ) {
		landmarkIdx.reserve(128);
		int idx;
		while(fin.good()) {
			fin >> idx;
			landmarkIdx.push_back(idx);
		}
		PhGUtils::message("landmarks loaded.");
		cout << "total landmarks = " << landmarkIdx.size() << endl;
		ndims_fpts = landmarkIdx.size() * 3;
	}
	else {
		PhGUtils::abort("Failed to load landmarks!");
	}
	// allocate space for landmarks
	checkCudaErrors(cudaMalloc((void**) &d_fptsIdx, sizeof(int)*landmarkIdx.size()));
	// upload the landmark indices
	checkCudaErrors(cudaMemcpy(d_fptsIdx, &(landmarkIdx[0]), sizeof(int)*landmarkIdx.size(), cudaMemcpyHostToDevice));

	h_q = new float[landmarkIdx.size()*3];
	checkCudaErrors(cudaMalloc((void**) &d_q, sizeof(float)*landmarkIdx.size()*3));
	h_q2d = new float[landmarkIdx.size()*3];
	checkCudaErrors(cudaMalloc((void**) &d_q2d, sizeof(float)*landmarkIdx.size()*3));

	ndims_pts = core_dim[2];	// constraints by the vertices, at most 3 constraints for each vertex

	checkCudaErrors(cudaMalloc((void**) &d_targetLocations, sizeof(float)*ndims_pts));
	showCUDAMemoryUsage();

	PhGUtils::message("allocating memory for computataion ...");
	// allocate space for Aid, Aexp, AidtAid, AexptAexp, brhs, Aidtb, Aexptb
	checkCudaErrors(cudaMalloc((void **) &d_RTparams, sizeof(float)*7));
	int maxParams = max(ndims_wid, ndims_wexp);
	checkCudaErrors(cudaMalloc((void **) &d_A, sizeof(float)*(maxParams + ndims_fpts + ndims_pts) * maxParams));
	checkCudaErrors(cudaMalloc((void **) &d_b, sizeof(float)*(ndims_pts + ndims_fpts + maxParams)));

	h_w_landmarks = new float[landmarkIdx.size()*3];
	checkCudaErrors(cudaMalloc((void**) &d_w_landmarks, sizeof(float)*landmarkIdx.size()*3));

	checkCudaErrors(cudaMalloc((void**) &d_icpc, sizeof(d_ICPConstraint)*MAX_ICPC_COUNT));
	checkCudaErrors(cudaMalloc((void**) &d_nicpc, sizeof(int)));
	PhGUtils::message("done.");

	PhGUtils::message("allocating memory for incoming data ...");
	checkCudaErrors(cudaMalloc((void**) &d_colordata, sizeof(unsigned char)*640*480*4));
	checkCudaErrors(cudaMalloc((void**) &d_depthdata, sizeof(unsigned char)*640*480*4));

	checkCudaErrors(cudaMalloc((void**) &d_indexMap, sizeof(unsigned char)*640*480*4));
	checkCudaErrors(cudaMalloc((void**) &d_depthMap, sizeof(float)*640*480));
	PhGUtils::message("done.");

	showCUDAMemoryUsage();
}

__host__ void MultilinearReconstructorGPU::bindTarget(const vector<PhGUtils::Point3f>& pts)
{
	cout << "binding " << pts.size() << " targets ..." << endl;
	// update q array and q2d array on host side
	// they are stored in page-locked memory
	int npts = pts.size();
	for(int i=0;i<npts;i++) {
		int idx = i*3;
		h_q2d[idx] = pts[i].x, h_q2d[idx+1] = pts[i].y, h_q2d[idx+2] = pts[i].z;
		PhGUtils::colorToWorld(pts[i].x, pts[i].y, pts[i].z, h_q[idx], h_q[idx+1], h_q[idx+2]);
	}

	// compute depth mean and variance
	int validZcount = 0;
	float mu_depth = 0, sigma_depth = 0;
	for(int i=0;i<npts;i++) {
		float z = pts[i].z;
		if( z != 0 ){
			mu_depth += z;
			validZcount++;
		}
	}
	mu_depth /= validZcount;
	for(int i=0;i<npts;i++) {
		float z = pts[i].z;
		if( z != 0 ){
			float dz = z - mu_depth;
			sigma_depth += dz * dz;
		}
	}
	sigma_depth /= (validZcount-1);

	const float DEPTH_THRES = 1e-6;
	int validCount = 0;
	meanX = 0; meanY = 0; meanZ = 0;
	// initialize weights
	for(int i=0, idx=0;i<npts;i++, idx+=3) {
		const float3& p = make_float3(h_q[idx], h_q[idx+1], h_q[idx+2]);
		int isValid = (fabs(p.z) > DEPTH_THRES)?1:0;

		meanX += p.x * isValid;
		meanY += p.y * isValid;
		meanZ += p.z * isValid;

		float dz = p.z - mu_depth;
		float w_depth = exp(-fabs(dz) / (sigma_depth*50.0));

		// set the landmark weights
		h_w_landmarks[idx] = h_w_landmarks[idx+1] = h_w_landmarks[idx+2] = (i<64 || i>74)?isValid*w_depth:isValid*w_boundary*w_depth;
		validCount += isValid;
	}

	// upload to GPU
	PhGUtils::message("uploading targets to GPU ...");
	checkCudaErrors(cudaMemcpy(d_q2d, h_q2d, sizeof(float)*npts*3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_q, h_q, sizeof(float)*npts*3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_w_landmarks, h_w_landmarks, sizeof(float)*npts*3, cudaMemcpyHostToDevice));
	PhGUtils::message("done.");
}

__host__ void MultilinearReconstructorGPU::bindRGBDTarget(const vector<unsigned char>& colordata,
														  const vector<unsigned char>& depthdata) 
{
	PhGUtils::message("uploading image targets to GPU ...");

	// update both color data and depth data
	const int sz = sizeof(unsigned char)*640*480*4;
	checkCudaErrors(cudaMemcpy(d_colordata, &(colordata[0]), sz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depthdata, &(depthdata[0]), sz, cudaMemcpyHostToDevice));

	PhGUtils::message("done.");
}

__host__ void MultilinearReconstructorGPU::setBaseMesh(const PhGUtils::QuadMesh& m) {
	baseMesh = m;
	// upload the mesh topology
	int nfaces = baseMesh.faceCount();
	vector<int4> topo(nfaces);
	for(int i=0;i<nfaces;i++) {
		const PhGUtils::QuadMesh::face_t& f = baseMesh.face(i);
		topo[i] = make_int4(f.x, f.y, f.z, f.w);
	}

	PhGUtils::message("uploading mesh topology");
	cout << "face count = " << nfaces << endl;
	if( d_meshtopo ) {
		checkCudaErrors(cudaFree(d_meshtopo));
	}
	checkCudaErrors(cudaMalloc((void**) &d_meshtopo, sizeof(int4)*nfaces));
	checkCudaErrors(cudaMemcpy(d_meshtopo, &(topo[0]), sizeof(int4)*nfaces, cudaMemcpyHostToDevice));
	showCUDAMemoryUsage();
}

__host__ void MultilinearReconstructorGPU::initRenderer() {
	// off-screen rendering related
	depthMap.resize(640*480);
	indexMap.resize(640*480*4);
	mProj = PhGUtils::KinectColorProjection.transposed();
	mMv = PhGUtils::Matrix4x4f::identity();

	dummyWgt = shared_ptr<QGLWidget>(new QGLWidget());
	dummyWgt->hide();
	dummyWgt->makeCurrent();
	fbo = shared_ptr<QGLFramebufferObject>(new QGLFramebufferObject(640, 480, QGLFramebufferObject::Depth));
	dummyWgt->doneCurrent();
}

__host__ void MultilinearReconstructorGPU::fit(FittingOption op) {
	switch( op ) {
	case FIT_POSE:
		{
			fitPose();
			break;
		}
	case FIT_IDENTITY:
		{

			break;
		}
	case FIT_EXPRESSION:
		{

			break;
		}
	case FIT_POSE_AND_IDENTITY:
		{

			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{

			break;
		}
	case FIT_ALL:
		{

			break;
		}
	}
}

__host__ void MultilinearReconstructorGPU::fitPose() {
	cout << "fitting pose ..." << endl;
	
	// make rotation matrix and translation vector
	cout << "initial guess ..." << endl;
	PhGUtils::printArray(h_RTparams, 7);

	float errorThreshold_ICP = 1e-5;
	float errorDiffThreshold_ICP = errorThreshold * 1e-4;

	int iters = 0;
	float E0 = 0, E;
	bool converged = false;
	const int MaxIterations = 32;

	while( !converged && iters++<MaxIterations ) {
		transformMesh();
		updateMesh();
		renderMesh();
		int nicpc = collectICPConstraints(iters, MaxIterations);
		converged = fitRigidTransformation();
		E = computeError();
		PhGUtils::debug("iters", iters, "Error", E);

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (nicpc/5000.0));
		converged |= fabs(E - E0) < errorDiffThreshold_ICP;
		E0 = E;
	}

	// use the latest parameters
	transformMesh();
	updateMesh();
}

__host__ void MultilinearReconstructorGPU::fitPoseAndIdentity() {
}

__host__ void MultilinearReconstructorGPU::fitPoseAndExpression() {
}

__host__ void MultilinearReconstructorGPU::fitAll() {
}

__host__ void MultilinearReconstructorGPU::renderMesh()
{
	dummyWgt->makeCurrent();
	fbo->bind();

#if FBO_DEBUG_GPU
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
#if FBO_DEBUG_GPU
	GLenum errcode = glGetError();
	if (errcode != GL_NO_ERROR) {
		const GLubyte *errString = gluErrorString(errcode);
		fprintf (stderr, "OpenGL Error: %s\n", errString);
	}
#endif

	glReadPixels(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, &(indexMap[0]));
#if FBO_DEBUG_GPU
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

#if FBO_DEBUG_GPU
	ofstream fout("fbodepth.txt");
	PhGUtils::print2DArray(&(depthMap[0]), 480, 640, fout);
	fout.close();

	QImage img = PhGUtils::toQImage(&(indexMap[0]), 640, 480);	
	img.save("fbo.png");
#endif

	// upload result to GPU
	checkCudaErrors(cudaMemcpy(d_indexMap, &indexMap[0], sizeof(unsigned char)*640*480*4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depthMap, &depthMap[0], sizeof(float)*640*480, cudaMemcpyHostToDevice));
}

__global__ void clearICPConstraints(int* nicpc) {
	*nicpc = 0;
}

__device__ __forceinline__ float3 color2world(float u, float v, float d) {
	// focal length
	const float fx_rgb = 525.0, fy_rgb = 525.0;
	// for 640x480 image
	const float cx_rgb = 320.0, cy_rgb = 240.0;

	// This part is correct now.
	// Given a Kinect depth value, its depth in OpenGL coordinates
	// system must be negative.
	float depth = -d/1000.0;

	float3 res;
	// inverse mapping of projection
	res.x = -(u - cx_rgb) * depth / fx_rgb;
	res.y = (v - cy_rgb) * depth / fy_rgb;
	res.z = depth;
	return res;
}

__device__ __forceinline__ float3 world2color(float3 p) {
	// focal length
	const float fx_rgb = 525.0, fy_rgb = 525.0;
	// for 640x480 image
	const float cx_rgb = 320.0, cy_rgb = 240.0;

	float invZ = 1.0 / p.z;
	float3 uvd;
	uvd.x = clamp(cx_rgb - p.x * fx_rgb * invZ, 0.f, 639.f);
	uvd.y = clamp(cy_rgb + p.y * fy_rgb * invZ, 0.f, 479.f);
	uvd.z = -p.z*1000.0f;
	return uvd;
}

__device__ __forceinline__ int decodeIndex(unsigned char r, unsigned char g, unsigned char b) {
	int ri = r, gi = g, bi = b;
	return (ri << 16) | (gi << 8) | bi;
}

__device__ float point_to_triangle_distance(float3 p0, float3 p1, float3 p2, float3 p3, float3& hit) {
	float dist = 0;

	float3 d = p1 - p0;
	float3 e12 = p2 - p1, e13 = p3 - p1, e21 = -e12, e23 = p3 - p2, e31 = -e13, e32 = -e23;
	float3 e12n = normalize(e12), e13n = normalize(e13), e21n = -e12n, e23n = normalize(e23), e31n = -e13n, e32n = -e23n;

	float3 n = normalize(cross(e12, e13));

	float dDOTn = dot(d, n);
	float dnorm = length(d);
	float cosAlpha = dDOTn / dnorm;

	float dn = dDOTn * cosAlpha;
	float3 p0p0c = -dn * n;
	float3 p0c = p0 + p0p0c;

	float3 v1 = e21n + e31n, v2 = e12n + e32n, v3 = e13n + e23n;
	float3 p0cp1 = p1 - p0c, p0cp2 = p2 - p0c, p0cp3 = p3 - p0c;

	float3 c1 = cross(p0cp1, p0cp2), c2 = cross(p0cp2, p0cp3), c3 = cross(p0cp3, p0cp1);

	float d1 = dot(c1, n);
	float d2 = dot(c2, n);
	float d3 = dot(c3, n);

	float3 x0 = p0c, x1, x2;
	bool inside = true;
	if ( d1 < d2 && d1 < d3 && d1<0 )			//-- outside, p1, p2 side
	{
		inside = false;		
		x1 = p1; x2 = p2;
	}
	else if ( d2 < d1 && d2 < d3 && d2<0 )	//-- outside, p2, p3 side
	{
		inside = false;
		x1 = p2; x2 = p3;
	}
	else if ( d3 < d1 && d3 < d2 && d3<0 )	//-- outside, p3, p1 side
	{
		inside = false;
		x1 = p3; x2 = p1;
	}

	if (inside)
	{
		hit = p0c;
		dist = dn;
	}
	else
	{
		float3 x1x0 = x0 - x1, x2x0 = x0 - x1, x2x1 = x1 - x2;
		float L_x2x0 = length(x2x0);
		float t = dot(x1x0, x2x0) / (L_x2x0 * L_x2x0);

		hit = x1 + t * x2x1;

		float3 line = p0 - hit;
		dist = length(line);
	}

	return dist;

}

__device__ float3 compute_barycentric_coordinates(float3 p, float3 q1, float3 q2, float3 q3) {
	float3 e23 = q3 - q2, e21 = q1 - q2, e31 = q1 - q3;
	float3 d2 = p - q2, d3 = p - q3;
	float3 oriN = cross(e23, e21);
	float3 n = normalize(oriN);

	float invBTN = 1.0 / dot(oriN, n);
	float3 bcoord;
	bcoord.x = dot(cross(e23, d2), n) * invBTN;
	bcoord.y = dot(cross(e31, d3), n) * invBTN;
	bcoord.z = 1 - bcoord.x - bcoord.y;

	return bcoord;
}
//@note	need to upload the topology of the template mesh for constraint collection
__global__ void collectICPConstraints_kernel(
						float*				mesh,
						int4*				meshtopo,
						unsigned char*		indexMap,			// synthesized data
						float*				depthMap,			// synthesized data
						unsigned char*		colordata,			// capture data
						unsigned char*		depthdata,			// capture data
						d_ICPConstraint*	icpc,				// ICP constraints
						int*				nicpc,
						float thres
	) {
	float DIST_THRES = thres;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x > 639 || y > 479 ) return;

	int tid = y * 640 + x;

	int u = x, v = y;
	int idx = (v * 640 + u)*4;
	int vv = 479 - y;
	int didx = vv * 640 + u;
	
	if( depthMap[didx] < 1.0 ) {
		// valid pixel, see if it is a valid constraint
		float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);
		
		// bad pixel
		if( d == 0 ) return;

		// compute target location
		float3 q = color2world(u, v, d);

		// take a small window
		const int wSize = 2;
		int checkedFaces[(wSize+1)*(wSize+1)];
		int checkedCount = 0;
		float closestDist = FLT_MAX;
		int3 closestVerts;
		float3 closestHit;

		// check for the closest point face
		for(int r = max(v - wSize, 0); r <= min(v + wSize, 479); r++) {
			int rr = 479 - r;
			for(int c = max(u - wSize, 0); c <= min(u + wSize, 479); c++) {
				int pidx = rr * 640 + c;
				int poffset = pidx << 2;

				float depthVal = depthMap[pidx];
				if( depthVal < 1.0 ) {
					int fidx = decodeIndex(indexMap[poffset], indexMap[poffset+1], indexMap[poffset+2]);

					
					bool checked = false;
					// see if this face is already checked
					for(int j=0;j<checkedCount;j++) {
						if( fidx == checkedFaces[j] ){
							checked = true;
							break;
						}
					}
					if( checked ) continue;
					else {
						checkedFaces[checkedCount] = fidx;
						checkedCount++;
					}


					// not checked yet, check out this face
					int4 f = meshtopo[fidx];
					int4 vidx = f * 3;
					float3 v0 = make_float3(mesh[vidx.x], mesh[vidx.x+1], mesh[vidx.x+2]);
					float3 v1 = make_float3(mesh[vidx.y], mesh[vidx.y+1], mesh[vidx.y+2]);
					float3 v2 = make_float3(mesh[vidx.z], mesh[vidx.z+1], mesh[vidx.z+2]);
					float3 v3 = make_float3(mesh[vidx.w], mesh[vidx.w+1], mesh[vidx.w+2]);

					float3 hit1, hit2;
					float dist1 = point_to_triangle_distance(q, v0, v1, v2, hit1);
					float dist2 = point_to_triangle_distance(q, v1, v2, v3, hit2);
				
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
			}
		}

		if( closestDist < DIST_THRES ) {
			d_ICPConstraint cc;
			cc.q = q;
			cc.v = closestVerts;
			int3 vidx = cc.v*3;
			
			float3 v0 = make_float3(mesh[vidx.x], mesh[vidx.x+1], mesh[vidx.x+2]);
			float3 v1 = make_float3(mesh[vidx.y], mesh[vidx.y+1], mesh[vidx.y+2]);
			float3 v2 = make_float3(mesh[vidx.z], mesh[vidx.z+1], mesh[vidx.z+2]);
			
			cc.bcoords = compute_barycentric_coordinates( closestHit, v0, v1, v2 );
			int slot = atomicAdd(nicpc, 1);
			__threadfence();
			icpc[slot] = cc;
		}
	}
}

__host__ int MultilinearReconstructorGPU::collectICPConstraints(int iters, int maxIters) {
	const float DIST_THRES_MAX = 0.010;
	const float DIST_THRES_MIN = 0.001;
	float DIST_THRES = DIST_THRES_MAX + (DIST_THRES_MIN - DIST_THRES_MAX) * iters / (float)maxIters;
	PhGUtils::message("Collecting ICP constraints...");
	
	writeback(d_depthMap, 480, 640, "d_depthmap.txt");

	clearICPConstraints<<<1, 1, 0, mystream>>>(d_nicpc);
	checkCudaState();
	PhGUtils::Timer ticpc;
	ticpc.tic();
	dim3 block(16, 16, 1);
	dim3 grid(640/16, 480/16, 1);
	collectICPConstraints_kernel<<<grid, block, 0, mystream>>>( d_mesh,
																d_meshtopo,
																d_indexMap,
																d_depthMap,
																d_colordata,
																d_depthdata,
																d_icpc,
																d_nicpc,
																DIST_THRES);

	cudaThreadSynchronize();
	ticpc.toc("ICPC collection");
	checkCudaState();
	PhGUtils::message("ICPC computed.");
	// copy back the number of ICP constraints
	int icpcCount = 0;
	checkCudaErrors(cudaMemcpy(&icpcCount, d_nicpc, sizeof(int), cudaMemcpyDeviceToHost));
	cout << "ICPC = " << icpcCount << endl;
	
#if OUTPUT_ICPC
	vector<d_ICPConstraint> icpc(640*480);
	checkCudaErrors(cudaMemcpy(&icpc[0], d_icpc, sizeof(d_ICPConstraint)*MAX_ICPC_COUNT, cudaMemcpyDeviceToHost));
	ofstream fout("d_icpc.txt");
	for(int i=0;i<icpcCount;i++) {
		float3 bc = icpc[i].bcoords;
		int3 vidx = icpc[i].v * 3;
		float3 p;
		p.x = tmesh(vidx.x  ) * bc.x + tmesh(vidx.y  ) * bc.y + tmesh(vidx.z  ) * bc.z;
		p.y = tmesh(vidx.x+1) * bc.x + tmesh(vidx.y+1) * bc.y + tmesh(vidx.z+1) * bc.z;
		p.z = tmesh(vidx.x+2) * bc.x + tmesh(vidx.y+2) * bc.y + tmesh(vidx.z+2) * bc.z;
		fout << icpc[i].q.x << ' '
			 << icpc[i].q.y << ' '
			 << icpc[i].q.z << ' '
			 << p.x << ' '
			 << p.y << ' '
			 << p.z << ' '
			 << bc.x << ' '
			 << bc.y << ' '
			 << bc.z << endl;
	}
	fout.close();
#endif

	return icpcCount;
}

// use one dimensional configuration
__global__ void cost_ICP(float *unknowns, float *costfunc, int offset,
							 d_ICPConstraint* d_icpc, int nicpc,
							 float *d_tplt,
							 float w_ICP
							 ) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nicpc ) return;

	float s, rx, ry, rz, tx, ty, tz;
	rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
	tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
	s = unknowns[6];

	mat3 R = mat3::rotation(rx, ry, rz) * s;
	float3 T = make_float3(tx, ty, tz);

	d_ICPConstraint icpc = d_icpc[tid];
	const int3& v = icpc.v;
	const float3& bc = icpc.bcoords;

	int3 vidx = icpc.v*3;

	float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
	float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
	float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);

	const float3& q = icpc.q;

	float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;
	p = R * p + T;

	costfunc[tid+offset] = length(p - q) * w_ICP;
}

__global__ void jacobian_ICP(float *unknowns, float *J, int offset,
							 d_ICPConstraint* d_icpc, int nicpc,
							 float *d_tplt,
							 float w_ICP) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nicpc ) return;

	float s, rx, ry, rz, tx, ty, tz;
	rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
	tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
	s = unknowns[6];

	mat3 R = mat3::rotation(rx, ry, rz) * s;
	float3 T = make_float3(tx, ty, tz);
	mat3 Jx, Jy, Jz;
	mat3::jacobian(rx, ry, rz, Jx, Jy, Jz);

	d_ICPConstraint icpc = d_icpc[tid];
	const int3& v = icpc.v;
	const float3& bc = icpc.bcoords;

	int3 vidx = icpc.v*3;

	float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
	float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
	float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);

	const float3& q = icpc.q;

	float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;

	int jidx = tid*7+offset;

	// R * p
	float3 rp = R * p;

	// s * R * p + t - q
	float3 rk = s * rp + T - q;

	float3 jp = Jx * p;
	// \frac{\partial r_i}{\partial \theta_x}
	J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;	

	jp = Jy * p;
	// \frac{\partial r_i}{\partial \theta_y}
	J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

	jp = Jz * p;
	// \frac{\partial r_i}{\partial \theta_z}
	J[jidx++] = 2.0 * s * dot(jp, rk) * w_ICP;

	// \frac{\partial r_i}{\partial \t_x}
	J[jidx++] = 2.0 * rk.x * w_ICP;

	// \frac{\partial r_i}{\partial \t_y}
	J[jidx++] = 2.0 * rk.y * w_ICP;

	// \frac{\partial r_i}{\partial \t_z}
	J[jidx++] = 2.0 * rk.z * w_ICP;

	// \frac{\partial r_i}{\partial s}
	J[jidx++] = 2.0 * dot(rp, rk) * w_ICP;
}

// use one dimensional configuration
/* @note	d_w_mask is		1.0				if i<42 || i > 74
							w_outer * w_fp	if 63 < i <= 74
							w_chin * w_fp	if 42 <= i <= 63
*/
__global__ void cost_FeaturePoints(float *unknowns, float *costfunc, int offset,
								   int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,
								   float *d_tplt,
								   float *d_w_landmarks, float *d_w_mask,
								   float w_fp_scale) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nfpts ) return;

	float s, rx, ry, rz, tx, ty, tz;
	rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
	tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
	s = unknowns[6];

	mat3 R = mat3::rotation(rx, ry, rz) * s;
	float3 T = make_float3(tx, ty, tz);

	int voffset = tid * 3;
	float wpt = d_w_landmarks[tid] * w_fp_scale * d_w_mask[tid];

	int vidx = d_fptsIdx[tid] * 3;
	float3 p = make_float3(d_tplt[vidx], d_tplt[vidx+1], d_tplt[vidx+2]);


	if( tid < 42 || tid > 74 ) {
		float3 q = make_float3(d_q[voffset], d_q[voffset+1], d_q[voffset+2]);
		costfunc[tid+offset] = length(p-q)*wpt;
	}
	else {
		float3 q = make_float3(d_q2d[voffset], d_q2d[voffset+1], d_q2d[voffset+2]);
		float3 uvd = world2color(p);
		float du = uvd.x - q.x, dv = uvd.y - q.y;
		costfunc[tid+offset] = (du*du+dv*dv)*wpt;
	}
}

__global__ void jacobian_FeaturePoints(float *unknowns, float *J, int offset,
								   int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,
								   float *d_tplt,
								   float *d_w_landmarks, float *d_w_mask,
								   float w_fp_scale) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nfpts ) return;

	float s, rx, ry, rz, tx, ty, tz;
	rx = unknowns[0], ry = unknowns[1], rz = unknowns[2];
	tx = unknowns[3], ty = unknowns[4], tz = unknowns[5];
	s = unknowns[6];

	mat3 R = mat3::rotation(rx, ry, rz) * s;
	float3 T = make_float3(tx, ty, tz);
	mat3 Jx, Jy, Jz;
	mat3::jacobian(rx, ry, rz, Jx, Jy, Jz);

	int voffset = tid * 3;
	float wpt = d_w_landmarks[tid] * w_fp_scale * d_w_mask[tid];

	int vidx = d_fptsIdx[tid] * 3;
	int jidx = tid*7+offset;
	float3 p = make_float3(d_tplt[vidx], d_tplt[vidx+1], d_tplt[vidx+2]);

	if( tid < 42 || tid > 74 ) {
		float3 q = make_float3(d_q[voffset], d_q[voffset+1], d_q[voffset+2]);

		// R * p
		float3 rp = R * p;

		// s * R * p + t - q
		float3 rk = s * rp + T - q;

		float3 jp = Jx * p;
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;	

		jp = Jy * p;
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;

		jp = Jz * p;
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * dot(jp, rk) * wpt;

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * rk.x * wpt;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * rk.y * wpt;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * rk.z * wpt;

		// \frac{\partial r_i}{\partial s}
		J[jidx++] = 2.0 * dot(rp, rk) * wpt;
	}
	else {
		float3 q = make_float3(d_q2d[voffset], d_q2d[voffset+1], d_q2d[voffset+2]);

		float3 rp = R * p;
		float3 pk = s * rp + T;

		float inv_z = 1.0 / pk.z;
		float inv_z2 = inv_z * inv_z;

		const float f_x = 525.0, f_y = 525.0;
		float Jf[6] = {0};
		Jf[0] = -f_x * inv_z; Jf[2] = f_x * pk.x * inv_z2;
		Jf[4] = f_y * inv_z; Jf[5] = -f_y * pk.y * inv_z2;

		float3 uvd = world2color(pk);
		float pu = uvd.x, pv = uvd.y, pd = uvd.z;

		// residue
		float rkx = pu - q.x, rky = pv - q.y;

		// J_? * p_k
		float3 jp = Jx * p;
		// J_f * J_? * p_k
		float jfjpx, jfjpy;
		
		jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
		jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
		// \frac{\partial r_i}{\partial \theta_x}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

		jp = Jy * p;
		jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
		jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
		// \frac{\partial r_i}{\partial \theta_y}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

		jp = Jz * p;
		jfjpx = Jf[0] * jp.x + Jf[2] * jp.z;
		jfjpy = Jf[4] * jp.y + Jf[5] * jp.z;
		// \frac{\partial r_i}{\partial \theta_z}
		J[jidx++] = 2.0 * s * (jfjpx * rkx + jfjpy * rky) * wpt;

		// \frac{\partial r_i}{\partial \t_x}
		J[jidx++] = 2.0 * (Jf[0] * rkx) * wpt;

		// \frac{\partial r_i}{\partial \t_y}
		J[jidx++] = 2.0 * (Jf[4] * rky) * wpt;

		// \frac{\partial r_i}{\partial \t_z}
		J[jidx++] = 2.0 * (Jf[2] * rkx + Jf[5] * rky) * wpt;

		// \frac{\partial r_i}{\partial s}
		jfjpx = Jf[0] * rp.x + Jf[2] * rp.z;
		jfjpy = Jf[4] * rp.y + Jf[5] * rp.z;
		J[jidx++] = 2.0 * (jfjpx * rkx + jfjpy * rky) * wpt;
	}
}

__global__ void cost_History() {
}

__global__ void jacobian_History() {
}

__host__ bool MultilinearReconstructorGPU::fitRigidTransformation() {
	// gauss-newton algorithm to estimate a new set of parameters
	int iters = GaussNewton();
	// update the parameters and check if convergence is obtained
	return true;
}

__global__ void computeError_kernel() {
}

__host__ float MultilinearReconstructorGPU::computeError() {
	return true;
}

// one dimensional configuration
// R is the rows of the rotation matrix
__device__ float3 R0, R1, R2, T;
__global__ void setupRigidTransformation(float r00, float r01, float r02,
										 float r10, float r11, float r12,
										 float r20, float r21, float r22,
										 float  t0, float  t1,  float t2)
{
	R0 = make_float3(r00, r01, r02);
	R1 = make_float3(r10, r11, r12);
	R2 = make_float3(r20, r21, r22);
	 T = make_float3( t0,  t1,  t2);

	printf("%f, %f, %f\n", R0.x, R0.y, R0.z);
	printf("%f, %f, %f\n", R1.x, R1.y, R1.z);
	printf("%f, %f, %f\n", R2.x, R2.y, R2.z);
}

__global__ void transformMesh_kernel(int nverts, float *d_tplt, float *d_mesh) 
{
	unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if( tid >= nverts ) return;

	unsigned int idx0 = tid*3;

	float3 p = make_float3(d_tplt[idx0], d_tplt[idx0+1], d_tplt[idx0+2]);

	d_mesh[idx0] = dot(R0, p) + T.x;
	d_mesh[idx0+1] = dot(R1, p) + T.y;
	d_mesh[idx0+2] = dot(R2, p) + T.z;
}

__host__ void MultilinearReconstructorGPU::transformMesh() {
	PhGUtils::Matrix3x3f Rot = PhGUtils::rotationMatrix(h_RTparams[0], h_RTparams[1], h_RTparams[2]) * h_RTparams[6];
	cout << Rot << endl;
	float3 Tvec = make_float3(h_RTparams[3], h_RTparams[4], h_RTparams[5]);
	cout << Tvec << endl;
	int npts = ndims_pts/3;
	
	checkCudaState();

	setupRigidTransformation<<<1, 1, 0, mystream>>>(Rot(0, 0), Rot(0, 1), Rot(0, 2), Rot(1, 0), Rot(1, 1), Rot(1, 2),
									   Rot(2, 0), Rot(2, 1), Rot(2, 2),	   Tvec.x,    Tvec.y,    Tvec.z);
	checkCudaState();

	cout << "npts = " << npts << endl;
	dim3 block(256, 1);
	dim3 grid((int)ceil(npts/(float)(block.x)), 1, 1);
	cout << "grid: " << grid.x << "x" << grid.y << endl;
	transformMesh_kernel<<<grid, block, 0, mystream>>>(npts, d_tplt, d_mesh);
	
	checkCudaState();
}

__host__ void MultilinearReconstructorGPU::updateMesh()
{
	cout << "mesh size = " << tmesh.length() << endl;
	cout << "device mesh address = " << d_mesh << endl;
	cout << "bytes to transfer = " << sizeof(float)*ndims_pts << endl;
	checkCudaErrors(cudaMemcpy(tmesh.rawptr(), d_mesh, sizeof(float)*ndims_pts, cudaMemcpyDeviceToHost));
	cudaError_t err = cudaDeviceSynchronize();
	checkCudaErrors(err);

	writeback(d_mesh, ndims_pts/3, 3, "d_mesh.txt");
	writeback(d_tplt, ndims_pts/3, 3, "d_tplt.txt");

	//#pragma omp parallel for
	for(int i=0;i<tmesh.length()/3;i++) {
		int idx = i * 3;
		baseMesh.vertex(i).x = tmesh(idx++);
		baseMesh.vertex(i).y = tmesh(idx++);
		baseMesh.vertex(i).z = tmesh(idx);
	}

#if 1
	PhGUtils::OBJWriter writer;
	writer.save(baseMesh, "../Data/tmesh.obj");
#endif
}
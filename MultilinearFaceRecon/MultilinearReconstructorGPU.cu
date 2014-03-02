#include "MultilinearReconstructorGPU.cuh"
#include <helper_math.h>
#include <helper_functions.h>
#include "Kinect/KinectUtils.h"
#include "Utils/Timer.h"
#include "Utils/stringutils.h"
#include "Utils/utility.hpp"

#include "Elements_GPU.h"
#include "utils_GPU.cuh"
#include "numerical_algorithms.cuh"


#define FBO_DEBUG_GPU 0
#define KERNEL_DEBUG 0
#define OUTPUT_ICPC 0
#define FAST_RENDER 1

MultilinearReconstructorGPU::MultilinearReconstructorGPU():
	d_tu0(nullptr), d_tu1(nullptr), d_tm0(nullptr), d_tm1(nullptr),
	d_tplt(nullptr), d_mesh(nullptr), d_tm0RT(nullptr), d_tm1RT(nullptr),
	d_fptsIdx(nullptr), d_q2d(nullptr), d_q(nullptr), 
	d_colordata(nullptr), d_depthdata(nullptr),
	d_targetLocations(nullptr), d_RTparams(nullptr),
	d_A(nullptr), d_b(nullptr), d_meshtopo(nullptr), d_meshverts(nullptr)
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

	meanX = meanY = meanZ = 0;

	// initialize offscreen renderer
	initRenderer();

	// should be large enough
	NumericalAlgorithms::initialize(50, 16384);
	
	// initialize members
	init();

	initializeWeights();

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
	PhGUtils::printArray(params, 7);
	checkCudaErrors(cudaMemcpy(d_RTparams, params, sizeof(float)*7, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_meanRT, params, sizeof(float)*7, cudaMemcpyHostToDevice));
}

__host__ void MultilinearReconstructorGPU::setIdentityWeights(const Tensor1<float>& t) {
	// copy to GPU
	checkCudaErrors(cudaMemcpy(d_Wid, t.rawptr(), sizeof(float)*ndims_wid, cudaMemcpyHostToDevice));
	// update tensor tm0
	cublasSgemv('N', ndims_wexp * ndims_pts, ndims_wid, 1.0, d_tu0, ndims_wexp * ndims_pts, d_Wid, 1, 0, d_tm0, 1);

	// update distance map
	cublasSgemv('N', npts_mesh, ndims_wid, 1.0, d_rawdistmap, npts_mesh, d_Wid, 1, 0, d_distmap, 1);
	checkCudaState();
	writeback(d_distmap, npts_mesh, "d_distmap.txt");
	writeback(d_rawdistmap, ndims_wid, npts_mesh, "d_rawdistmap.txt");

	// and the template mesh
	cublasSgemv('T', ndims_wid, ndims_pts, 1.0, d_tm1, ndims_wid, d_Wid, 1, 0.0, d_tplt, 1);
}

__host__ void MultilinearReconstructorGPU::setExpressionWeights(const Tensor1<float>& t) {
	t.print();
	// copy to GPU
	checkCudaErrors(cudaMemcpy(d_Wexp, t.rawptr(), sizeof(float)*ndims_wexp, cudaMemcpyHostToDevice));
	// update tensor tm1
	cublasSgemv('N', ndims_wid * ndims_pts, ndims_wexp, 1.0, d_tu1, ndims_wid * ndims_pts, d_Wexp, 1, 0, d_tm1, 1);

	// and the template mesh
	cublasSgemv('T', ndims_wid, ndims_pts, 1.0, d_tm1, ndims_wid, d_Wid, 1, 0.0, d_tplt, 1);
}

__host__ void MultilinearReconstructorGPU::preprocess() {
	PhGUtils::message("preprocessing the input data...");

	// process distance map
	cublasSgemv('N', npts_mesh, ndims_wid, 1.0, d_rawdistmap, ndims_pts, d_mu_wid0, 1, 0, d_distmap, 1);

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
	npts_mesh = core_dim[2]/3;

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

	Tensor2<float> distmap;
	distmap.read("../Data/blendshape/distmap.bin");
	checkCudaErrors(cudaMalloc((void **) &d_rawdistmap, sizeof(float)*npts_mesh*core_dim[0]));
	checkCudaErrors(cudaMemcpy(d_rawdistmap, distmap.rawptr(), sizeof(float)*npts_mesh*core_dim[0], cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_distmap, sizeof(float)*npts_mesh));
	showCUDAMemoryUsage();
	writeback(d_rawdistmap, core_dim[0], npts_mesh, "d_rawdistmap.txt");

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
		nfpts = landmarkIdx.size();
	}
	else {
		PhGUtils::abort("Failed to load landmarks!");
	}
	// allocate space for landmarks
	checkCudaErrors(cudaMalloc((void**) &d_fptsIdx, sizeof(int)*landmarkIdx.size()));
	// upload the landmark indices
	checkCudaErrors(cudaMemcpy(d_fptsIdx, &(landmarkIdx[0]), sizeof(int)*landmarkIdx.size(), cudaMemcpyHostToDevice));

	/*
	h_q = new float[landmarkIdx.size()*3];
	checkCudaErrors(cudaMalloc((void**) &d_q, sizeof(float)*landmarkIdx.size()*3));
	h_q2d = new float[landmarkIdx.size()*3];
	checkCudaErrors(cudaMalloc((void**) &d_q2d, sizeof(float)*landmarkIdx.size()*3));
	*/
	checkCudaErrors(cudaHostAlloc((void**) &h_q, sizeof(float)*landmarkIdx.size()*3, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_q, h_q, 0));
	checkCudaErrors(cudaHostAlloc((void**) &h_q2d, sizeof(float)*landmarkIdx.size()*3, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_q2d, h_q2d, 0));

	ndims_pts = core_dim[2];	// constraints by the vertices, at most 3 constraints for each vertex

	checkCudaErrors(cudaMalloc((void**) &d_targetLocations, sizeof(float)*ndims_pts));
	showCUDAMemoryUsage();

	PhGUtils::message("allocating memory for computataion ...");
	// allocate space for Aid, Aexp, AidtAid, AexptAexp, brhs, Aidtb, Aexptb
	checkCudaErrors(cudaMalloc((void **) &d_RTparams, sizeof(float)*7));
	checkCudaErrors(cudaMalloc((void **) &d_meanRT, sizeof(float)*7));

	int maxParams = max(ndims_wid, ndims_wexp);
	checkCudaErrors(cudaMalloc((void **) &d_A, sizeof(double)*(maxParams + ndims_fpts + ndims_pts) * maxParams));
	checkCudaErrors(cudaMalloc((void **) &d_b, sizeof(double)*(ndims_pts + ndims_fpts + maxParams)));

	checkCudaErrors(cudaMalloc((void **) &d_AtA, sizeof(double)*maxParams*maxParams));
	checkCudaErrors(cudaMalloc((void **) &d_Atb, sizeof(double)*maxParams));

	/*
	h_w_mask = new float[landmarkIdx.size()];
	checkCudaErrors(cudaMalloc((void**) &d_w_mask, sizeof(float)*landmarkIdx.size()));

	h_w_landmarks = new float[landmarkIdx.size()];
	checkCudaErrors(cudaMalloc((void**) &d_w_landmarks, sizeof(float)*landmarkIdx.size()));
	*/
	checkCudaErrors(cudaHostAlloc((void**) &h_w_mask, sizeof(float)*landmarkIdx.size(), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_w_mask, h_w_mask, 0));
	checkCudaErrors(cudaHostAlloc((void**) &h_w_landmarks, sizeof(float)*landmarkIdx.size(), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**) &d_w_landmarks, h_w_landmarks, 0));


	checkCudaErrors(cudaMalloc((void**) &d_icpc, sizeof(d_ICPConstraint)*MAX_ICPC_COUNT));
	checkCudaErrors(cudaMalloc((void**) &d_nicpc, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &d_icpc_rigid, sizeof(d_ICPConstraint)*MAX_ICPC_COUNT));
	checkCudaErrors(cudaMalloc((void**) &d_nicpc_rigid, sizeof(int)));


	h_error = new float[MAX_ICPC_COUNT];
	checkCudaErrors(cudaMalloc((void**) &d_error, sizeof(float)*MAX_ICPC_COUNT));
	h_w_error = new float[MAX_ICPC_COUNT];
	checkCudaErrors(cudaMalloc((void**) &d_w_error, sizeof(float)*MAX_ICPC_COUNT));
	PhGUtils::message("done.");

	PhGUtils::message("allocating memory for incoming data ...");
	checkCudaErrors(cudaMalloc((void**) &d_colordata, sizeof(unsigned char)*640*480*4));
	checkCudaErrors(cudaMalloc((void**) &d_depthdata, sizeof(unsigned char)*640*480*4));

	checkCudaErrors(cudaMalloc((void**) &d_indexMap, sizeof(unsigned char)*640*480*4));
	checkCudaErrors(cudaMalloc((void**) &d_depthMap, sizeof(float)*640*480));
	PhGUtils::message("done.");

	showCUDAMemoryUsage();
}

__host__ void MultilinearReconstructorGPU::initializeWeights() {
	// read the weights from a setting file
	string fweights = "../Data/weights.txt";
	ifstream fin(fweights, ios::in);
	if( !fin ){
		PhGUtils::fail("failed to load weights file. using default weights");
		w_prior_id = 1e-3;
		w_prior_exp = 7.5e-4;

		w_boundary = 1e-8;
		w_chin = 5e-6;
		w_outer = 2.5e-10;
		w_fp = 0.25;

		w_history = 0.01;
		w_ICP = 1.0;		
	}
	else {
		fin >> w_prior_id >> w_prior_exp
			>> w_boundary >> w_chin >> w_outer
			>> w_fp >> w_ICP >> w_history;
	}
	fin.close();


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

	for(int i=0;i<78;i++) {
		if( i < 42 || i > 74 ) h_w_mask[i] = w_fp;
		else {
			if( i > 63 ) h_w_mask[i] = w_outer;
			else h_w_mask[i] = w_chin;
		}
	}
	//checkCudaErrors(cudaMemcpy(d_w_mask, w_mask, sizeof(float)*78, cudaMemcpyHostToDevice));

	totalCons = 0;
}

__host__ void MultilinearReconstructorGPU::bindTarget(const vector<PhGUtils::Point3f>& pts)
{
	//cout << "binding " << pts.size() << " targets ..." << endl;
	// update q array and q2d array on host side
	// they are stored in page-locked memory
	int numpts = pts.size();
	for(int i=0;i<numpts;i++) {
		int idx = i*3;
		h_q2d[idx] = pts[i].x, h_q2d[idx+1] = pts[i].y, h_q2d[idx+2] = pts[i].z;
		PhGUtils::colorToWorld(pts[i].x, pts[i].y, pts[i].z, h_q[idx], h_q[idx+1], h_q[idx+2]);
	}

	// compute depth mean and variance
	int validZcount = 0;
	float mu_depth = 0, sigma_depth = 0;
	for(int i=0;i<numpts;i++) {
		float z = h_q[i*3+2];
		if( z != 0 ){
			mu_depth += z;
			validZcount++;
		}
	}
	mu_depth /= validZcount;
	for(int i=0;i<numpts;i++) {
		float z = h_q[i*3+2];
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
	for(int i=0, idx=0;i<numpts;i++, idx+=3) {
		const float3& p = make_float3(h_q[idx], h_q[idx+1], h_q[idx+2]);
		int isValid = (fabs(p.z) > DEPTH_THRES)?1:0;

		meanX += p.x * isValid;
		meanY += p.y * isValid;
		meanZ += p.z * isValid;

		float dz = p.z - mu_depth;
		float w_depth = exp(-fabs(dz) / (sigma_depth*100.0));

		// set the landmark weights
		h_w_landmarks[i] = (i<64 || i>74)?isValid*w_depth:1.0;
		validCount += isValid;
	}

	// upload to GPU
	/*
	//PhGUtils::message("uploading targets to GPU ...");
	cudaMemcpy(d_q2d, h_q2d, sizeof(float)*numpts*3, cudaMemcpyHostToDevice);
	checkCudaState();
	//writeback(d_q2d, numpts, 3, "d_q2d.txt");
	cudaMemcpy(d_q, h_q, sizeof(float)*numpts*3, cudaMemcpyHostToDevice);
	checkCudaState();
	//writeback(d_q, numpts, 3, "d_q.txt");
	cudaMemcpy(d_w_landmarks, h_w_landmarks, sizeof(float)*numpts, cudaMemcpyHostToDevice);
	checkCudaState();
	//PhGUtils::message("done.");
	*/
}

__host__ void MultilinearReconstructorGPU::bindRGBDTarget(const vector<unsigned char>& colordata,
														  const vector<unsigned char>& depthdata) 
{
	//PhGUtils::message("uploading image targets to GPU ...");

	// update both color data and depth data
	const int sz = sizeof(unsigned char)*640*480*4;
	cudaMemcpy(d_colordata, &(colordata[0]), sz, cudaMemcpyHostToDevice);
	checkCudaState();
	cudaMemcpy(d_depthdata, &(depthdata[0]), sz, cudaMemcpyHostToDevice);
	checkCudaState();

	//PhGUtils::message("done.");
}

__host__ void MultilinearReconstructorGPU::setBaseMesh(const PhGUtils::QuadMesh& m) {
	baseMesh = m;
	// upload the mesh topology
	int nfaces = baseMesh.faceCount();
	int nverts = baseMesh.vertCount();
	cout << "setting base mesh: #v = " << nverts << ", #f = " << nfaces << endl;
	validfaces = 0;
	frontFaces.clear();
	vector<int4> topo(nfaces);
	//h_meshtopo.resize(nfaces*2*3);
	isBackFace.resize(nfaces);
	h_meshverts.resize(nfaces*4);
	h_faceidx.resize(nfaces*4);
	for(int i=0;i<nfaces;i++) {
		const PhGUtils::QuadMesh::face_t& f = baseMesh.face(i);
		topo[i] = make_int4(f.x, f.y, f.z, f.w);
		const PhGUtils::QuadMesh::vert_t& v0 = baseMesh.vertex(f.x);
		if( v0.z < -0.75 ) isBackFace[i] = true;
		else isBackFace[i] = false;
		//h_meshtopo[i*6+0] = f.x; h_meshtopo[i*6+1] = f.y; h_meshtopo[i*6+2] = f.z;
		//h_meshtopo[i*6+3] = f.y; h_meshtopo[i*6+4] = f.z; h_meshtopo[i*6+5] = f.w;

		float3 clr;
		PhGUtils::encodeIndex<float>(i, clr.x, clr.y, clr.z);		
		// fill the color array		
#if FAST_RENDER
		if( isBackFace[i] ) continue;
		else frontFaces.push_back(i);
		h_faceidx[validfaces+0] = h_faceidx[validfaces+1] = h_faceidx[validfaces+2] = h_faceidx[validfaces+3] = clr;
        validfaces+=4;
#else
		h_faceidx[f.x] = clr;
		h_faceidx[f.y] = clr;
		h_faceidx[f.z] = clr;
		h_faceidx[f.w] = clr;
#endif
	}

	PhGUtils::message("uploading mesh topology");
	cout << "face count = " << nfaces << endl;
	if( d_meshtopo ) {
		checkCudaErrors(cudaFree(d_meshtopo));
	}
	checkCudaErrors(cudaMalloc((void**) &d_meshtopo, sizeof(int4)*nfaces));
	checkCudaErrors(cudaMemcpy(d_meshtopo, &(topo[0]), sizeof(int4)*nfaces, cudaMemcpyHostToDevice));

	if( d_meshverts ) {
		checkCudaErrors(cudaFree(d_meshverts));
	}
	checkCudaErrors(cudaMalloc((void**) &d_meshverts, sizeof(float3)*nfaces*4));

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
			fitPoseAndIdentity();
			break;
		}
	case FIT_POSE_AND_EXPRESSION:
		{
			fitPoseAndExpression();
			break;
		}
	case FIT_ALL:
		{

			break;
		}
	}

	// store pose history
	if( useHistory ) {
		// post process, impose a moving average for pose
		RTHistory.push_back(vector<float>(h_RTparams, h_RTparams+7));
		if( RTHistory.size() > historyLength ) RTHistory.pop_front();
		vector<float> m = computeWeightedMeanPose();

		cudaMemcpy(d_meanRT, &(m[0]), sizeof(float)*7, cudaMemcpyHostToDevice);
		checkCudaState();
	}
}

__host__ void MultilinearReconstructorGPU::fitPose() {
	//cout << "fitting pose ..." << endl;
	
	// make rotation matrix and translation vector
	//cout << "initial guess ..." << endl;
	//PhGUtils::printArray(h_RTparams, 7);

	cc = 1e-4;
	float errorThreshold_ICP = 1e-5;
	float errorDiffThreshold_ICP = errorThreshold * 1e-4;

	int iters = 0;
	float E0 = 0, E;
	bool converged = false;
	const int MaxIterations = 64;

	int rigidIters;

	while( !converged && iters++<MaxIterations ) {
		transformMesh();
		updateMesh();
		renderMesh();
		nicpc = collectICPConstraints(iters, MaxIterations);
		converged = fitRigidTransformation(true, rigidIters);
		E = computeError();
		//PhGUtils::debug("iters", iters, "Error", E);

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (nicpc/5000.0));
		converged |= fabs(E - E0) < errorDiffThreshold_ICP;
		E0 = E;
	}

	// use the latest parameters
	transformMesh();
	updateMesh();
}

__global__ void transformTM0_kernel(float *d_tm0RT, mat3 R, int npts, int ndims) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= npts ) return;

	int stride = npts * 3;
	int offset  = tid * 3;
	for(int i=0;i<ndims;++i,offset+=stride) {
		rotate_point(R, d_tm0RT[offset], d_tm0RT[offset+1], d_tm0RT[offset+2]);
	}
}

__host__ void MultilinearReconstructorGPU::transformTM0() {
	cudaMemcpy(d_tm0RT, d_tm0, sizeof(float)*ndims_pts*ndims_wexp, cudaMemcpyDeviceToDevice);
	checkCudaState();
	// call the transformation kernel
	const int threads = 1024;
	mat3 R = mat3::rotation(h_RTparams[0], h_RTparams[1], h_RTparams[2])*h_RTparams[6];
	transformTM0_kernel<<<(int)(ceil(npts_mesh/(float)threads)), threads>>>(d_tm0RT, R, npts_mesh, ndims_wexp);
	checkCudaState();
	//writeback(d_tm0RT, ndims_wexp, npts_mesh*3, "d_tm0RT.txt");
	//checkCudaState();
}

__global__ void transformTM1_kernel(float *d_tm1RT, mat3 R, int npts, int ndims) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= npts ) return;
	int offset  = tid * 3 * ndims;
	for(int i=0;i<ndims;++i,offset++) {
		rotate_point(R, d_tm1RT[offset], d_tm1RT[offset+ndims], d_tm1RT[offset+ndims*2]);
	}
}

__host__ void MultilinearReconstructorGPU::transformTM1() {
	cudaMemcpy(d_tm1RT, d_tm1, sizeof(float)*ndims_pts*ndims_wid, cudaMemcpyDeviceToDevice);
	checkCudaState();
	// call the transformation kernel
	const int threads = 1024;
	mat3 R = mat3::rotation(h_RTparams[0], h_RTparams[1], h_RTparams[2])*h_RTparams[6];
	transformTM1_kernel<<<(int)(ceil(npts_mesh/(float)threads)), threads>>>(d_tm1RT, R, npts_mesh, ndims_wid);
	checkCudaState();
	//writeback(d_tm1RT, npts_mesh*3, ndims_wid, "d_tm1RT.txt");
	//checkCudaState();
}

__global__ void fitIdentity_ICPCTerm(d_ICPConstraint *d_icpc, int nicpc, int ndims, int off, int boff,
									 float *d_tm1RT, double *d_A, double *d_b,
									 float3 T, float w_ICP) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nicpc ) return;

	const d_ICPConstraint& icpc = d_icpc[tid];
	const float3& bc = icpc.bcoords;
	int3 vidx = icpc.v * 3 * ndims;
	int ndims2 = ndims*2;

	int offset = tid * 3 * ndims + off;
	for(int i=0;i<ndims;++i) {
		int3 k = vidx + i;
		float3 v0 = make_float3(d_tm1RT[k.x], d_tm1RT[k.x+ndims], d_tm1RT[k.x+ndims2]);
		float3 v1 = make_float3(d_tm1RT[k.y], d_tm1RT[k.y+ndims], d_tm1RT[k.y+ndims2]);
		float3 v2 = make_float3(d_tm1RT[k.z], d_tm1RT[k.z+ndims], d_tm1RT[k.z+ndims2]);
		float3 p = (v0 * bc.x + v1 * bc.y + v2 * bc.z) * w_ICP;

		int j = offset + i;
		d_A[j] = p.x;
		d_A[j+ndims] = p.y;
		d_A[j+ndims2] = p.z;
	}

	const float3& q = icpc.q;
	int boffset = tid * 3 + boff;
	d_b[boffset  ] = (q.x - T.x) * w_ICP;
	d_b[boffset+1] = (q.y - T.y) * w_ICP;
	d_b[boffset+2] = (q.z - T.z) * w_ICP;
}

__global__ void fitIdentity_FeaturePointsTerm(int *d_fptsIdx, float *d_q, int nfpts, int ndims, int off, int boff,
											  float *d_tm1RT, double *d_A, double *d_b,											  
											  float3 T, float *d_w_landmarks, float w_fp_scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nfpts ) return;

	int offset = tid * 3 * ndims + off;
	float wpt = d_w_landmarks[tid] * w_fp_scale;
	int voffset = d_fptsIdx[tid] * 3 * ndims;
	int ndims2 = ndims*2;

	for(int i=0;i<ndims;++i) {
		int j = offset+i;
		int k = voffset+i;
		float3 p = make_float3(d_tm1RT[k], d_tm1RT[k+ndims], d_tm1RT[k+ndims2]);	
		d_A[j] = p.x * wpt;
		d_A[j+ndims] = p.y * wpt;
		d_A[j+ndims2] = p.z * wpt;
	}

	int boffset = tid * 3 + boff;
	int qoffset = tid * 3;

	if( d_q[qoffset+2] == 0 ) wpt = 0;

	d_b[boffset  ] = (d_q[qoffset  ] - T.x) * wpt;
	d_b[boffset+1] = (d_q[qoffset+1] - T.y) * wpt;
	d_b[boffset+2] = (d_q[qoffset+2] - T.z) * wpt;
}

__global__ void fitIdentity_PriorTerm(double *d_A, double *d_b, float *d_sigma_wid_weighted, float *d_mu_wid_weighted,
									  int ndims, int off, int boff, 
									  float w_prior_scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= ndims ) return;

	int poffset = tid*ndims;
	int offset = poffset+off;	
	for(int i=0;i<ndims;i++) {		
		d_A[offset++] = d_sigma_wid_weighted[poffset++] * w_prior_scale;
	}

	d_b[tid+boff] = d_mu_wid_weighted[tid] * w_prior_scale;
}

__global__ void copySolutionToB(double *d_Atb, float *d_b, int ndims)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= ndims ) return;

	const float alpha = 1.0;
	d_b[tid] = alpha * d_Atb[tid] + (1.0 - alpha) * d_b[tid];
}

__host__ bool MultilinearReconstructorGPU::fitIdentityWeights() {
	float3 T = make_float3(h_RTparams[3], h_RTparams[4], h_RTparams[5]);
	const int threads = 1024;
	checkCudaState();
	// assemble the matrix and right hand side
	fitIdentity_ICPCTerm<<<(int)(ceil(nicpc/(float)threads)),threads>>>(d_icpc, nicpc, ndims_wid, 0, 0,
									 d_tm1RT, d_A, d_b,
									 T, w_ICP);
	checkCudaState();
	//cout << nicpc << ", " << ndims_wid << endl;
	//writeback(d_A, nicpc*3, ndims_wid, "d_A0.txt");
	//writeback(d_b, nicpc*3, 1, "d_b0.txt");
	checkCudaState();
	fitIdentity_FeaturePointsTerm<<<1, 128>>>(d_fptsIdx, d_q, nfpts, ndims_wid, nicpc*ndims_wid*3, nicpc*3,
											  d_tm1RT, d_A, d_b,											  
											  T, d_w_landmarks, w_fp);
	checkCudaState();
	//writeback(d_A, (nicpc+nfpts)*3, ndims_wid, "d_A1.txt");
	//writeback(d_b, (nicpc+nfpts)*3, 1, "d_b1.txt");
	checkCudaState();
	fitIdentity_PriorTerm<<<1, 64>>>(d_A, d_b, d_sigma_wid, d_mu_wid_weighted,
									  ndims_wid, (nicpc+nfpts)*ndims_wid*3, (nicpc+nfpts)*3, 
									  w_prior_id);
	checkCudaState();
	//writeback(d_A, (nicpc+nfpts)*3+ndims_wid, ndims_wid, "d_A2.txt");
	//writeback(d_b, (nicpc+nfpts)*3+ndims_wid, 1, "d_b2.txt");
	checkCudaState();

	vector<float> Wid(ndims_wid), brhs(ndims_wid);
	cudaMemcpy(&Wid[0], d_Wid, sizeof(float)*ndims_wid, cudaMemcpyDeviceToHost);
	checkCudaState();

	// solve for new set of parameters
#if USE_LS_SOLVER
	culaStatus s = culaDeviceSgels('T', ndims_wid, (nicpc+nfpts)*3+ndims_wid, 1, d_A, ndims_wid, d_b, (nicpc+nfpts)*3+ndims_wid);
	if( s != culaNoError ) {
		cerr << "cula failed!" << endl;
		if( s == culaArgumentError )
			printf("Argument %d has an illegal value\n", culaGetErrorInfo());
		else if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n", culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));		
	}

	cudaMemcpy(d_Wid, d_b, sizeof(float)*ndims_wid, cudaMemcpyDeviceToDevice);
	checkCudaState();
#else
	// normal mat method
	cublasDsyrk('U', 'N', ndims_wid, (nicpc+nfpts)*3+ndims_wid, 1.0, d_A, ndims_wid, 0.0, d_AtA, ndims_wid);
	//writeback(d_AtA, ndims_wid, ndims_wid, "d_AtA.txt");
	cublasDgemv('N', ndims_wid, (nicpc+nfpts)*3+ndims_wid, 1.0, d_A, ndims_wid, d_b, 1, 0.0, d_Atb, 1);
	//writeback(d_Atb, ndims_wid, 1, "d_Atb.txt");

	culaDeviceDpotrf('U', ndims_wid, d_AtA, ndims_wid);
	culaDeviceDpotrs('U', ndims_wid, 1, d_AtA, ndims_wid, d_Atb, ndims_wid);

	copySolutionToB<<<1, 64>>>(d_Atb, d_Wid, ndims_wid);
	//writeback(d_Wid, ndims_wid, 1, "d_wid.txt");
	checkCudaState();
#endif

	cudaMemcpy(&brhs[0], d_b, sizeof(float)*ndims_wid, cudaMemcpyDeviceToHost);
	checkCudaState();

	float diff = 0;
	for(int i=0;i<ndims_wid;i++) {
		diff += fabs(Wid[i] - brhs[i]);
		//cout << Wid[i] << "\t" << brhs[i] << endl;
	}
	//cout << endl;

	return diff/ndims_wid < cc;
}

__host__ void MultilinearReconstructorGPU::fitPoseAndIdentity() {
	cc = 1e-4;
	float errorThreshold_ICP = 1e-5;
	float errorDiffThreshold_ICP = errorThreshold * 1e-4;

	int iters = 0;
	float E0 = 0, E;
	bool converged = false;
	const int MaxIterations = 64;
	int rigidIters;

	while( !converged && iters++<MaxIterations ) {
		converged = true;
		transformMesh();
		updateMesh();
		renderMesh();
		nicpc = collectICPConstraints(iters, MaxIterations);
		converged &= fitRigidTransformation(true, rigidIters);

		// transform tm1
		transformTM1();

		// fit identity weights
		converged &= fitIdentityWeights();
		
		// update tplt with tm1
		cublasSgemv('T', ndims_wid, ndims_pts, 1.0, d_tm1, ndims_wid, d_Wid, 1, 0.0, d_tplt, 1);
		//writeback(d_tplt, npts_mesh, 3, "d_tplt.txt");

		// update distance map
		cublasSgemv('N', npts_mesh, ndims_wid, 1.0, d_rawdistmap, npts_mesh, d_Wid, 1, 0, d_distmap, 1);
		checkCudaState();
		//writeback(d_distmap, npts_mesh, "d_distmap.txt");
		//writeback(d_rawdistmap, ndims_wid, npts_mesh, "d_rawdistmap.txt");
		//::system("pause");

		E = computeError();
		PhGUtils::debug("iters", iters, "Error", E, "Error diff", fabs(E-E0));

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (nicpc/5000.0));
		converged |= fabs(E - E0) < errorDiffThreshold_ICP;
		E0 = E;
	}

	// use the latest parameters
	transformMesh();
	updateMesh();
		
	// update tm0, for the following steps
	cublasSgemv('N', ndims_wexp * ndims_pts, ndims_wid, 1.0, d_tu0, ndims_wexp * ndims_pts, d_Wid, 1, 0, d_tm0, 1);

	// update distance map
	cublasSgemv('N', npts_mesh, ndims_wid, 1.0, d_rawdistmap, npts_mesh, d_Wid, 1, 0, d_distmap, 1);
}

__global__ void fitExpression_ICPCTerm(d_ICPConstraint *d_icpc, int nicpc, int npts, int ndims, int off, int boff,
									 float *d_tm1RT, double *d_A, double *d_b,
									 float3 T, float w_ICP) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nicpc ) return;

	const d_ICPConstraint& icpc = d_icpc[tid];
	const float3& bc = icpc.bcoords;
	int3 vidx = icpc.v * 3;

	int offset = tid * 3 * ndims + off;
	int boffset = tid * 3 + boff;
	int stride = npts * 3;
	for(int i=0;i<ndims;++i) {
		int3 j = vidx + i * stride;
		float3 v0 = make_float3(d_tm1RT[j.x], d_tm1RT[j.x+1], d_tm1RT[j.x+2]);
		float3 v1 = make_float3(d_tm1RT[j.y], d_tm1RT[j.y+1], d_tm1RT[j.y+2]);
		float3 v2 = make_float3(d_tm1RT[j.z], d_tm1RT[j.z+1], d_tm1RT[j.z+2]);
		float3 p = (v0 * bc.x + v1 * bc.y + v2 * bc.z) * w_ICP;

		d_A[offset+i] = p.x;
		d_A[offset+i+ndims] = p.y;
		d_A[offset+i+ndims*2] = p.z;
	}

	const float3& q = icpc.q;

	d_b[boffset  ] = (q.x - T.x) * w_ICP;
	d_b[boffset+1] = (q.y - T.y) * w_ICP;
	d_b[boffset+2] = (q.z - T.z) * w_ICP;
}

__global__ void fitExpression_FeaturePointsTerm(int *d_fptsIdx, float *d_q, int nfpts, int npts, int ndims, int off, int boff,
											  float *d_tm1RT, double *d_A, double *d_b,											  
											  float3 T, float *d_w_landmarks, float w_fp_scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nfpts ) return;

	int offset = tid * 3 * ndims + off;
	float wpt = d_w_landmarks[tid] * w_fp_scale;
	int voffset = d_fptsIdx[tid] * 3;
	int stride = npts * 3;
	for(int i=0;i<ndims;++i) {
		int j = offset+i;
		int k = voffset+i * stride;
		float3 p = make_float3(d_tm1RT[k], d_tm1RT[k+1], d_tm1RT[k+2]);	
		d_A[j] = p.x * wpt;
		d_A[j+ndims] = p.y * wpt;
		d_A[j+ndims*2] = p.z * wpt;
	}

	int boffset = tid * 3 + boff;
	int qoffset = tid * 3;

	if( d_q[qoffset+2] == 0 ) wpt = 0;

	d_b[boffset  ] = (d_q[qoffset  ] - T.x) * wpt;
	d_b[boffset+1] = (d_q[qoffset+1] - T.y) * wpt;
	d_b[boffset+2] = (d_q[qoffset+2] - T.z) * wpt;
}

__global__ void fitExpression_PriorTerm(double *d_A, double *d_b, float *d_sigma_wid_weighted, float *d_mu_wid_weighted,
									  int ndims, int off, int boff, 
									  float w_prior_scale) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= ndims ) return;

	int poffset = tid*ndims;
	int offset = poffset+off;	
	for(int i=0;i<ndims;i++) {		
		d_A[offset++] = d_sigma_wid_weighted[poffset++] * w_prior_scale;
	}

	d_b[tid+boff] = d_mu_wid_weighted[tid] * w_prior_scale;
}

__host__ bool MultilinearReconstructorGPU::fitExpressionWeights() {
	float3 T = make_float3(h_RTparams[3], h_RTparams[4], h_RTparams[5]);
	const int threads = 1024;
	checkCudaState();
	// assemble the matrix and right hand side
	fitExpression_ICPCTerm<<<(int)(ceil(nicpc/(float)threads)),threads>>>(d_icpc, nicpc, npts_mesh, ndims_wexp, 0, 0,
									 d_tm0RT, d_A, d_b,
									 T, w_ICP);
	checkCudaState();
	//cout << nicpc << ", " << ndims_wid << endl;
	//writeback(d_A, nicpc*3, ndims_wexp, "d_Aexp0.txt");
	checkCudaState();
	//cout << w_fp_scale << endl;
	const float w_fp = 5.0;
	fitExpression_FeaturePointsTerm<<<1, 128>>>(d_fptsIdx, d_q, nfpts, npts_mesh, ndims_wexp, nicpc*ndims_wexp*3, nicpc*3,
											  d_tm0RT, d_A, d_b,											  
											  T, d_w_landmarks, w_fp);
	checkCudaState();
	//writeback(d_A, (nicpc+nfpts)*3, ndims_wexp, "d_Aexp1.txt");
	checkCudaState();
	const float w_p = 2.5;
	fitExpression_PriorTerm<<<1, 64>>>(d_A, d_b, d_sigma_wexp, d_mu_wexp_weighted,
									  ndims_wexp, (nicpc+nfpts)*ndims_wexp*3, (nicpc+nfpts)*3, 
									  w_prior_exp);
	checkCudaState();
	//writeback(d_A, (nicpc+nfpts)*3+ndims_wexp, ndims_wexp, "d_Aexp2.txt");
	//checkCudaState();

	vector<float> Wexp(ndims_wexp), brhs(ndims_wexp);
	cudaMemcpy(&Wexp[0], d_Wexp, sizeof(float)*ndims_wexp, cudaMemcpyDeviceToHost);
	checkCudaState();

#if USE_LS_SOLVER
	// solve for new set of parameters
	culaStatus s = culaDeviceSgels('T', ndims_wexp, (nicpc+nfpts)*3+ndims_wexp, 1, d_A, ndims_wexp, d_b, (nicpc+nfpts)*3+ndims_wexp);
	if( s != culaNoError ) {
		cerr << "cula failed!" << endl;
		if( s == culaArgumentError )
			printf("Argument %d has an illegal value\n", culaGetErrorInfo());
		else if( s == culaDataError )
			printf("Data error with code %d, please see LAPACK documentation\n", culaGetErrorInfo());
		else
			printf("%s\n", culaGetStatusString(s));
	}

	cudaMemcpy(d_Wexp, d_b, sizeof(float)*ndims_wexp, cudaMemcpyDeviceToDevice);
	checkCudaState();

#else
	// normal mat method
	cublasDsyrk('U', 'N', ndims_wexp, (nicpc+nfpts)*3+ndims_wexp, 1.0, d_A, ndims_wexp, 0.0, d_AtA, ndims_wexp);
	//writeback(d_AtA, ndims_wexp, ndims_wexp, "d_AtA.txt");
	cublasDgemv('N', ndims_wexp, (nicpc+nfpts)*3+ndims_wexp, 1.0, d_A, ndims_wexp, d_b, 1, 0.0, d_Atb, 1);
	//writeback(d_Atb, ndims_wexp, 1, "d_Atb.txt");

	culaDeviceDpotrf('U', ndims_wexp, d_AtA, ndims_wexp);
	culaDeviceDpotrs('U', ndims_wexp, 1, d_AtA, ndims_wexp, d_Atb, ndims_wexp);

	copySolutionToB<<<1, 64>>>(d_Atb, d_Wexp, ndims_wexp);
	//writeback(d_Wexp, ndims_wexp, 1, "d_wexp.txt");
	checkCudaState();
#endif

	// full update
	cudaMemcpy(&brhs[0], d_b, sizeof(float)*ndims_wexp, cudaMemcpyDeviceToHost);
	checkCudaState();

	float diff = 0;
	//b.print("b");
	for(int i=0;i<ndims_wexp;i++) {
		diff += fabs(Wexp[i] - brhs[i]);
		//cout << '#' << i << ": " << Wexp[i] << '\t' << brhs[i] << endl;
	}

	return diff/ndims_wexp < cc;
}

__host__ void MultilinearReconstructorGPU::fitPoseAndExpression() {
	cc = 1e-4;
	float errorThreshold_ICP = 1e-5;
	float errorDiffThreshold_ICP = 1e-4;

	int iters = 0;
	float E0 = 1, E = 0;
	bool converged = false;
	const int MaxIterations = 16;

	constraintCount.clear();
	int rigidIters;

	while( !converged && iters++<MaxIterations ) {
		converged = true;
		tTrans.tic();
		transformMesh();
		tTrans.toc();

		tUpdate.tic();
		updateMesh();
		tUpdate.toc();

		tRender.tic();
		renderMesh();
		tRender.toc();

		tCollect.tic();
		nicpc = collectICPConstraints(iters, MaxIterations);
		constraintCount.push_back(nicpc);
		tCollect.toc();

		tRigid.tic();
		converged &= fitRigidTransformation(false, rigidIters);
		rigidIterations.push_back(rigidIters);
		tRigid.toc();

		tTrans0.tic();
		transformTM0();
		tTrans0.toc();

		tExpr.tic();
		converged &= fitExpressionWeights();
		tExpr.toc();

		tUpdate0.tic();
		cublasSgemv('N', ndims_pts, ndims_wexp, 1.0, d_tm0, ndims_pts, d_Wexp, 1, 0.0, d_tplt, 1);
		tUpdate0.toc();
		
		//::system("pause");
		tError.tic();
		E = computeError();
		tError.toc();
		//PhGUtils::debug("iters", iters, "Error", E, "Error diff", fabs((E-E0)/E0));

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (nicpc/5000.0));
		converged |= fabs((E-E0)/E0) < errorDiffThreshold_ICP;
		E0 = E;
	}

	/*
	ofstream fout;
	fout.open("d_wexp.txt", ios::app);
	vector<float> Wexp(ndims_wexp);
	cudaMemcpy(&Wexp[0], d_Wexp, sizeof(float)*ndims_wexp, cudaMemcpyDeviceToHost);
	for(int i=0;i<ndims_wexp;i++) 
		fout << Wexp[i] << ' ';
	fout << endl;
	fout.close();
	*/

	/*
	for(int i=0;i<7;i++)
		cout << h_RTparams[i] << '\t';
	cout << endl;
	*/
	double avgcons = PhGUtils::average(constraintCount);
	double avgiters = PhGUtils::average(rigidIterations);
	printf("Finished in %d iterations. \n \
		Average constaints = %8.1f. \n \
		Averate iterations for rigid transformation = %8.1f", iters, avgcons, avgiters);
	totalCons += avgcons;
	totalRigidIters += avgiters;
	constraintCount.clear();
	// use the latest parameters
	transformMesh();
	updateMesh();
}

__host__ void MultilinearReconstructorGPU::fitAll() {
	throw "Not implemented yet";
	cc = 1e-4;
	float errorThreshold_ICP = 1e-5;
	float errorDiffThreshold_ICP = errorThreshold * 1e-4;

	int iters = 0;
	float E0 = 0, E;
	bool converged = false;
	const int MaxIterations = 64;

	int rigidIters;

	while( !converged && iters++<MaxIterations ) {
		transformMesh();
		updateMesh();
		renderMesh();
		nicpc = collectICPConstraints(iters, MaxIterations);
		converged = fitRigidTransformation(true, rigidIters);
		E = computeError();
		//PhGUtils::debug("iters", iters, "Error", E);

		// adaptive threshold
		converged |= E < (errorThreshold_ICP / (nicpc/5000.0));
		converged |= fabs(E - E0) < errorDiffThreshold_ICP;
		E0 = E;
	}

	// use the latest parameters
	transformMesh();
	updateMesh();
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

#if !FAST_RENDER
	baseMesh.drawFaceIndices();
#else
	//Enable the vertex array functionality:
	glEnableClientState(GL_VERTEX_ARRAY);
	//Enable the color array functionality (so we can specify a color for each vertex)
	glEnableClientState(GL_COLOR_ARRAY);
	//pass the vertex pointer:
	glVertexPointer( 3,   //3 components per vertex (x,y,z)
					 GL_FLOAT,
					 sizeof(float3),
					 &h_meshverts[0]);
	//pass the color pointer
	glColorPointer(  3,   //3 components per vertex (r,g,b)
					 GL_FLOAT,
					 sizeof(float3),
					 &h_faceidx[0]);  //Pointer to the first color
	//cout << h_meshtopo.size() / 3 << endl;
	glDrawArrays( GL_QUADS, 0, validfaces );
	//glDrawElements(GL_TRIANGLES, h_meshtopo.size(), GL_UNSIGNED_INT, &h_meshtopo[0]);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
#endif

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
	::system("pause");
#endif

	// upload result to GPU
	checkCudaErrors(cudaMemcpy(d_indexMap, &indexMap[0], sizeof(unsigned char)*640*480*4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depthMap, &depthMap[0], sizeof(float)*640*480, cudaMemcpyHostToDevice));
}

__global__ void clearICPConstraints(int* nicpc, int *nicpc_rigid) {
	*nicpc = 0;
	*nicpc_rigid = 0;
}

//@note	need to upload the topology of the template mesh for constraint collection
__global__ void collectICPConstraints_kernel(
						float*				mesh,
						int4*				meshtopo,
						float*				d_distmap,
						unsigned char*		indexMap,			// synthesized data
						float*				depthMap,			// synthesized data
						unsigned char*		colordata,			// capture data
						unsigned char*		depthdata,			// capture data
						d_ICPConstraint*	icpc,				// ICP constraints
						int*				nicpc,
						d_ICPConstraint*    icpc_rigid,
						int*				nicpc_rigid,
						float thres
	) {
	float DIST_THRES = thres;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x > 639 || y > 479 ) return;

	int tid = y * 640 + x;

	//if( tid & 0x1 ) return;

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
		float3 q = color2world_fast(u, v, d);

		// take a small window
		const int wSize = 5;
		//int checkedFaces[9];
		//int checkedCount = 0;
		float closestDist = FLT_MAX;
		int3 closestVerts;
		float3 closestHit;

		// check for the closest point face
		for(int r = max(v - wSize, 0); r <= min(v + wSize, 479); r++) {
			int rr = 479 - r;
			for(int c = max(u - wSize, 0); c <= min(u + wSize, 639); c++) {
				int pidx = rr * 640 + c;
				int poffset = pidx << 2;

				float depthVal = depthMap[pidx];
				if( depthVal < 1.0 ) {
					int fidx = decodeIndex(indexMap[poffset], indexMap[poffset+1], indexMap[poffset+2]);
					fidx = clamp(fidx, 0, 11399);
					//bool checked = false;
					//// see if this face is already checked
					//for(int j=0;j<checkedCount;j++) {
					//	if( fidx == checkedFaces[j] ){
					//		checked = true;
					//		break;
					//	}
					//}
					//if( checked ) continue;
					//else {
					//	checkedFaces[checkedCount] = fidx;
					//	checkedCount++;
					//}


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
			cc.weight = mean(cc.bcoords * make_float3(d_distmap[cc.v.x], d_distmap[cc.v.y], d_distmap[cc.v.z]));

			int slot = atomicAdd(nicpc, 1);
			__threadfence();
			icpc[slot] = cc;

			const float cutoff = 1e-3;
			if( cc.weight < cutoff ) {
				int slot_rigid = atomicAdd(nicpc_rigid, 1);
				__threadfence();
				icpc_rigid[slot_rigid] = cc;
			}
		}
	}
}

__host__ int MultilinearReconstructorGPU::collectICPConstraints(int iters, int maxIters) {
	const float DIST_THRES_MAX = 0.010;
	const float DIST_THRES_MIN = 0.001;
	float DIST_THRES = DIST_THRES_MAX + (DIST_THRES_MIN - DIST_THRES_MAX) * iters / (float)maxIters;
	//PhGUtils::message("Collecting ICP constraints...");
	
	//writeback(d_depthMap, 480, 640, "d_depthmap.txt");

	clearICPConstraints<<<1, 1, 0, mystream>>>(d_nicpc, d_nicpc_rigid);
	checkCudaState();
	PhGUtils::Timer ticpc;
	//ticpc.tic();
	dim3 block(8, 8, 1);
	dim3 grid(640/block.x, 480/block.y, 1);
	collectICPConstraints_kernel<<<grid, block, 0, mystream>>>( d_mesh,
																d_meshtopo,
																d_distmap,
																d_indexMap,
																d_depthMap,
																d_colordata,
																d_depthdata,
																d_icpc,
																d_nicpc,
																d_icpc_rigid,
																d_nicpc_rigid,
																DIST_THRES);

	cudaThreadSynchronize();
	//ticpc.toc("ICPC collection");
	checkCudaState();
	//PhGUtils::message("ICPC computed.");
	// copy back the number of ICP constraints
	int icpcCount = 0;
	cudaMemcpy(&icpcCount, d_nicpc, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaState();
	//cout << "ICPC = " << icpcCount << endl;

	cudaMemcpy(&nicpc_rigid, d_nicpc_rigid, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaState();
	//cout << "ICPC_rigid = " << nicpc_rigid << endl;

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
			 << bc.z << ' '
			 << icpc[i].weight
			 << endl;
	}
	fout.close();
	::system("pause");
#endif

	return icpcCount;
}

__host__ vector<float> MultilinearReconstructorGPU::computeWeightedMeanPose() {
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

__host__ bool MultilinearReconstructorGPU::fitRigidTransformation(bool fitScale, int& iters) {
	int nparams = fitScale?7:6;

	d_ICPConstraint *icpc_ptr;
	int icpccount;

	if( fitScale ) {
		icpc_ptr = d_icpc;
		icpccount = nicpc;
	}
	else {
		icpc_ptr = d_icpc_rigid;
		icpccount = nicpc_rigid;
	}

	cudaMemcpy(NumericalAlgorithms::x, d_RTparams, sizeof(float)*7, cudaMemcpyDeviceToDevice);
	checkCudaState();
	int itmax = 32;
	float opts[] = {0.125, 1e-6, 1e-4};
	// gauss-newton algorithm to estimate a new set of parameters
	iters = NumericalAlgorithms::lm(
		nparams, nfpts+nicpc_rigid, itmax, opts,
		d_fptsIdx, d_q, d_q2d, nfpts, d_w_landmarks, d_w_mask, w_fp,
		icpc_ptr, icpccount, w_ICP,
		d_meanRT, w_history,
		d_tplt,
		mystream
		);
	cudaThreadSynchronize();
	PhGUtils::message("rigid transformation estimated in " + PhGUtils::toString(iters) + " iterations.");
	// update the parameters and check if convergence is obtained
	cudaMemcpy(d_RTparams, NumericalAlgorithms::x, sizeof(float)*nparams, cudaMemcpyDeviceToDevice);
	checkCudaState();
	vector<float> RTparams(nparams);
	cudaMemcpy(&(RTparams[0]), NumericalAlgorithms::x, sizeof(float)*nparams, cudaMemcpyDeviceToHost);
	checkCudaState();

	//PhGUtils::message("gauss-newton returned.");
	float diff = 0;
	for(int i=0;i<nparams;i++) {
		diff += fabs(RTparams[i] - h_RTparams[i]);
		h_RTparams[i] = RTparams[i];
		//cout << RTparams[i] << ' ';
	}
	//cout << endl;
	//::system("pause");

	return diff/nparams<cc || iters == 0;
}

__global__ void computeError_ICP(float *params, float *d_error, float *d_w_error, int offset, 
								 d_ICPConstraint *d_icpc, int nicpc, float w_ICP, float *d_tplt) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= nicpc ) return;

	float s, rx, ry, rz, tx, ty, tz;
	rx = params[0], ry = params[1], rz = params[2];
	tx = params[3], ty = params[4], tz = params[5];
	s = params[6];

	mat3 R = mat3::rotation(rx, ry, rz) * s;
	float3 T = make_float3(tx, ty, tz);

	int3 v = d_icpc[tid].v;
	float3 bc = d_icpc[tid].bcoords;

	int3 vidx = v * 3;

	float3 v0 = make_float3(d_tplt[vidx.x], d_tplt[vidx.x+1], d_tplt[vidx.x+2]);
	float3 v1 = make_float3(d_tplt[vidx.y], d_tplt[vidx.y+1], d_tplt[vidx.y+2]);
	float3 v2 = make_float3(d_tplt[vidx.z], d_tplt[vidx.z+1], d_tplt[vidx.z+2]);
	float3 p = v0 * bc.x + v1 * bc.y + v2 * bc.z;

	const float3& q = d_icpc[tid].q;

	// p = R * p + T
	p = R * p + T;

	d_w_error[tid+offset] = w_ICP;
	d_error[tid+offset] = dot(p-q, p-q) * w_ICP;
}

__global__ void computeError_FeaturePoints(float *params, float *d_error, float *d_w_error, int offset,
		int *d_fptsIdx, float *d_q, float *d_q2d, int nfpts,
		float *d_tplt,
		float *d_w_landmarks, float *d_w_mask,
		float w_fp_scale) 
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if( tid >= nfpts ) return;

		float s, rx, ry, rz, tx, ty, tz;
		rx = params[0], ry = params[1], rz = params[2];
		tx = params[3], ty = params[4], tz = params[5];
		s = params[6];

		mat3 R = mat3::rotation(rx, ry, rz) * s;
		float3 T = make_float3(tx, ty, tz);

		int voffset = tid * 3;
		float wpt = d_w_landmarks[tid] * w_fp_scale * d_w_mask[tid];

		int vidx = d_fptsIdx[tid] * 3;
		float3 p = make_float3(d_tplt[vidx], d_tplt[vidx+1], d_tplt[vidx+2]);
		p = R * p + T;


		if( tid < 42 || tid > 74 ) {
			float3 q = make_float3(d_q[voffset], d_q[voffset+1], d_q[voffset+2]);
			d_error[tid+offset] = dot(p-q, p-q)*wpt;
			d_w_error[tid+offset] = wpt;
		}
		else {
			float3 q = make_float3(d_q2d[voffset], d_q2d[voffset+1], d_q2d[voffset+2]);
			float3 uvd = world2color(p);
			float du = uvd.x - q.x, dv = uvd.y - q.y;
			d_error[tid+offset] = (du*du+dv*dv)*wpt;
			d_w_error[tid+offset] = wpt;
		}
	}

__host__ float MultilinearReconstructorGPU::computeError() {
	checkCudaState();
	//cout << d_error << endl;
	//cout << d_w_error << endl;
	computeError_ICP<<<dim3((int)(ceil(nicpc/1024.0)), 1, 1), dim3(1024, 1, 1), 0, mystream>>>(d_RTparams, d_error, d_w_error, 0, d_icpc, nicpc, w_ICP / nicpc, d_tplt);
	checkCudaState();
	computeError_FeaturePoints<<<dim3(1, 1, 1), dim3(nfpts, 1, 1), 0, mystream>>>(d_RTparams, d_error, d_w_error, nicpc, d_fptsIdx, d_q, d_q2d, nfpts, d_tplt, d_w_landmarks, d_w_mask, w_fp);
	checkCudaState();

	cudaMemcpy(h_error, d_error, sizeof(float)*(nicpc+nfpts), cudaMemcpyDeviceToHost);
	checkCudaState();
	//writeback(d_error, nicpc+nfpts, 1, "d_error.txt");
	cudaMemcpy(h_w_error, d_w_error, sizeof(float)*(nicpc+nfpts), cudaMemcpyDeviceToHost);
	checkCudaState();
	float E = 0, Wsum = 0;
	for(int i=0;i<nfpts+nicpc;++i) { E += h_error[i]; Wsum += h_w_error[i]; }

	return E / Wsum;
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

	//printf("%f, %f, %f\n", R0.x, R0.y, R0.z);
	//printf("%f, %f, %f\n", R1.x, R1.y, R1.z);
	//printf("%f, %f, %f\n", R2.x, R2.y, R2.z);
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
	//cout << Rot << endl;
	float3 Tvec = make_float3(h_RTparams[3], h_RTparams[4], h_RTparams[5]);
	//cout << Tvec << endl;
	int npts = ndims_pts/3;
	
	checkCudaState();

	setupRigidTransformation<<<1, 1, 0, mystream>>>(Rot(0, 0), Rot(0, 1), Rot(0, 2), Rot(1, 0), Rot(1, 1), Rot(1, 2),
									   Rot(2, 0), Rot(2, 1), Rot(2, 2),	   Tvec.x,    Tvec.y,    Tvec.z);
	checkCudaState();

	//cout << "npts = " << npts << endl;
	dim3 block(256, 1);
	dim3 grid((int)ceil(npts/(float)(block.x)), 1, 1);
	//cout << "grid: " << grid.x << "x" << grid.y << endl;
	transformMesh_kernel<<<grid, block, 0, mystream>>>(npts, d_tplt, d_mesh);
	
	checkCudaState();
}

__global__ void updateMesh_kernel(float* d_mesh, float3* d_meshverts, int4* d_meshtopo, int nfaces) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if( tid >= nfaces ) return;
	
	const int4& vidx = d_meshtopo[tid] * 3;
	int voffset = tid * 4;
	// fill the vertex array
	d_meshverts[voffset+0] = make_float3(d_mesh[vidx.x], d_mesh[vidx.x+1], d_mesh[vidx.x+2]);
	d_meshverts[voffset+1] = make_float3(d_mesh[vidx.y], d_mesh[vidx.y+1], d_mesh[vidx.y+2]);
	d_meshverts[voffset+2] = make_float3(d_mesh[vidx.z], d_mesh[vidx.z+1], d_mesh[vidx.z+2]);
	d_meshverts[voffset+3] = make_float3(d_mesh[vidx.w], d_mesh[vidx.w+1], d_mesh[vidx.w+2]);
}

__host__ void MultilinearReconstructorGPU::updateMesh()
{
	cudaMemcpy(tmesh.rawptr(), d_mesh, sizeof(float)*ndims_pts, cudaMemcpyDeviceToHost);
	checkCudaState();

	//#pragma omp parallel for
#if !FAST_RENDER
	//cout << "mesh size = " << tmesh.length() << endl;
	//cout << "device mesh address = " << d_mesh << endl;
	//cout << "bytes to transfer = " << sizeof(float)*ndims_pts << endl;

	//writeback(d_mesh, ndims_pts/3, 3, "d_mesh.txt");
	//writeback(d_tplt, ndims_pts/3, 3, "d_tplt.txt");

	for(int i=0;i<tmesh.length()/3;i++) {
		int idx = i * 3;
		baseMesh.vertex(i).x = tmesh(idx++);
		baseMesh.vertex(i).y = tmesh(idx++);
		baseMesh.vertex(i).z = tmesh(idx);
	}
#else
	for(int i=0, validfaces=0;i<frontFaces.size();i++) {
		const PhGUtils::QuadMesh::face_t& f = baseMesh.face(frontFaces[i]);
		int4 vidx = make_int4(f.x, f.y, f.z, f.w)*3;		
		float3 v0 = make_float3(tmesh(vidx.x), tmesh(vidx.x+1), tmesh(vidx.x+2));
		float3 v1 = make_float3(tmesh(vidx.y), tmesh(vidx.y+1), tmesh(vidx.y+2));
		float3 v2 = make_float3(tmesh(vidx.z), tmesh(vidx.z+1), tmesh(vidx.z+2));
		float3 v3 = make_float3(tmesh(vidx.w), tmesh(vidx.w+1), tmesh(vidx.w+2));
		// fill the vertex array
		h_meshverts[validfaces] = v0;
		h_meshverts[validfaces+1] = v1;
		h_meshverts[validfaces+2] = v2;
		h_meshverts[validfaces+3] = v3;
		validfaces+=4;
	}
	
	/*
	updateMesh_kernel<<<(int)ceil(baseMesh.faceCount()/1024.0), 1024>>>(d_mesh, d_meshverts, d_meshtopo, baseMesh.faceCount());
	checkCudaState();
	cudaMemcpy(&h_meshverts[0], d_meshverts, sizeof(float3)*baseMesh.faceCount()*4, cudaMemcpyDeviceToHost);
	checkCudaState();
	*/
#endif
	

#if 0
	PhGUtils::OBJWriter writer;
	writer.save(baseMesh, "../Data/tmesh.obj");
#endif
}

__host__ void MultilinearReconstructorGPU::printStats() {
	PhGUtils::message("Time cost for transforming mesh = " + PhGUtils::toString(tTrans.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for updating mesh = " + PhGUtils::toString(tUpdate.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for rendering mesh = " + PhGUtils::toString(tRender.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for collecting constraints = " + PhGUtils::toString(tCollect.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for estimating rigid transformation = " + PhGUtils::toString(tRigid.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for transforming template = " + PhGUtils::toString(tTrans0.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for estimating expression weights = " + PhGUtils::toString(tExpr.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for updating template = " + PhGUtils::toString(tUpdate0.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Time cost for computing error = " + PhGUtils::toString(tError.elapsed()*1000.0) + "ms.");
	PhGUtils::message("Total number of constraints = " + PhGUtils::toString(totalCons) + ".");
	PhGUtils::message("Total number of rigid tranformation iterations = " + PhGUtils::toString(totalRigidIters) + ".");
}
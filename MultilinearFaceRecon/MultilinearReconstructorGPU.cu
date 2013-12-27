#include "MultilinearReconstructorGPU.cuh"

MultilinearReconstructorGPU::MultilinearReconstructorGPU() {
	w_prior_id = 1e-3;
	w_prior_exp = 1e-4;
	w_boundary = 1e-6;

	meanX = meanY = meanZ = 0;

	culaInitialize();

	// initialize members
	init();

	// process the loaded data
	preprocess();
}

MultilinearReconstructorGPU::~MultilinearReconstructorGPU() {
	// release resources
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
	cublasSgemv('N', ndims_wexp * ndims_pts, ndims_wid, 1.0, d_tu0, ndims_wid, d_Wid, 1, 0, d_tm0, 1);

	// tm1 = tu1 * Wexp, use cublas
	// tu1: ndims_wexp * (ndims_wid * ndims_pts) matrix, each row corresponds to an expression
	//		inside each row, the vertices are arraged using index-major
	//		That is, a row in tu1 can be see as a column-major matrix where each column corresponds to an identity
	// tm1: a column-major matrix where each column corresponds to an identity
	cublasSgemv('N', ndims_wid * ndims_pts, ndims_wexp, 1.0, d_tu1, ndims_wexp, d_Wid, 1, 0, d_tm0, 1);

	// create template mesh
	// tplt = tm1 * Wid, use cublas

	// initialize tm0c, tm1c
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

	Tensor2<float> tu0 = core.unfold(0), tu1 = core.unfold(1);

	PhGUtils::message("transferring the unfolded core tensor to GPU ...");
	// transfer the unfolded core tensor to GPU
	checkCudaErrors(cudaMalloc((void **) &d_tu0, sizeof(float)*totalSize));
	checkCudaErrors(cudaMemcpy(d_tu0, tu0.rawptr(), sizeof(float)*totalSize, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_tu1, sizeof(float)*totalSize));
	checkCudaErrors(cudaMemcpy(d_tu1, tu1.rawptr(), sizeof(float)*totalSize, cudaMemcpyHostToDevice));

	PhGUtils::message("done.");
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
	}
	else {
		PhGUtils::abort("Failed to load landmarks!");
	}
	ndims_pts = landmarkIdx.size() * 3;	// constraints by the points

	// allocate space for landmarks
	checkCudaErrors(cudaMalloc((void**) &d_fptsIdx, sizeof(int)*landmarkIdx.size()));
	checkCudaErrors(cudaMalloc((void**) &d_targets, sizeof(float)*ndims_pts));

	// build the truncated tensor
	Tensor3<float> corec(core.dim(0), core.dim(1), ndims_pts);

	for(int i=0;i<core.dim(0);i++) {
		for(int j=0;j<core.dim(1);j++) {
			for(int k=0, idx=0;k<landmarkIdx.size();k++, idx+=3) {
				int vidx = landmarkIdx[k] * 3;
				corec(i, j, idx) = core(i, j, vidx);
				corec(i, j, idx+1) = core(i, j, vidx+1);
				corec(i, j, idx+2) = core(i, j, vidx+2);
			}
		}
	}
	int truncatedSize = core_dim[0] * core_dim[1] * ndims_pts;
	Tensor2<float> tu0c = corec.unfold(0), tu1c = corec.unfold(1);

	PhGUtils::message("transferring the truncated, unfolded core tensor to GPU ...");
	// transfer the unfolded core tensor to GPU
	checkCudaErrors(cudaMalloc((void **) &d_tu0c, sizeof(float)*truncatedSize));
	checkCudaErrors(cudaMemcpy(d_tu0c, tu0c.rawptr(), sizeof(float)*truncatedSize, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **) &d_tu1c, sizeof(float)*truncatedSize));
	checkCudaErrors(cudaMemcpy(d_tu1c, tu1c.rawptr(), sizeof(float)*truncatedSize, cudaMemcpyHostToDevice));

	PhGUtils::message("done.");
	showCUDAMemoryUsage();

	PhGUtils::message("allocating memory for computataion ...");
	// allocate space for Aid, Aexp, AidtAid, AexptAexp, brhs, Aidtb, Aexptb
	checkCudaErrors(cudaMalloc((void**) &d_RTparams, sizeof(float)*7));
	checkCudaErrors(cudaMalloc((void **) &d_Aid, sizeof(float)*(ndims_wid + ndims_pts)*ndims_wid));
	checkCudaErrors(cudaMalloc((void **) &d_Aexp, sizeof(float)*(ndims_wexp + ndims_pts)*ndims_wexp));
	checkCudaErrors(cudaMalloc((void **) &d_brhs, sizeof(float)*(ndims_pts + max(ndims_wid, ndims_wexp))));
	checkCudaErrors(cudaMalloc((void **) &d_w_landmarks, sizeof(float)*512));
	PhGUtils::message("done.");
}

__host__ void MultilinearReconstructorGPU::bindTarget(const vector<PhGUtils::Point3f>& tgt) {

}

__host__ bool MultilinearReconstructorGPU::fitPose() {
	return true;
}

__host__ bool MultilinearReconstructorGPU::fitPoseAndIdentity() {
	return true;
}

__host__ bool MultilinearReconstructorGPU::fitPoseAndExpression() {
	return true;
}

__host__ bool MultilinearReconstructorGPU::fitAll() {
	return true;
}
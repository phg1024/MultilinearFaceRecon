/* Gaussian-Newton algorithm for estimating rigid transformation */
__host__ int GaussNewton(
	float *x0, float *x, float *deltaX, float *r, float *J,
	float *JtJ, 
	int itmax, float *opts
	) 
{	
	// setup parameters
	float delta, R_THRES, DIFF_THRES;
	if( opts == NULL ) {
		// use default values
		delta = 1.0;	// step size, default to use standard Newton-Ralphson
		R_THRES = 1e-6;	DIFF_THRES = 1e-6;
	}
	else {
		delta = opts[0]; R_THRES = opts[1]; DIFF_THRES = opts[2];
	}

	// compute initial residue with GPU

	int iters = 0;

	// while not converged
	while( iters < itmax ) {
		//// compute jacobian with GPU

		//// store old values

		//// compute JtJ

		//// compute Jtr


		//// compute deltaX


		//// update x
	
	
		//// update residue

		iters++;
	}

	return iters;
}
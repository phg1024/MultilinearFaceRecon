#ifndef AAM_Search_REALGLOBAL_CUDA_H
#define AAM_Search_REALGLOBAL_CUDA_H

#include <string>
#include <fstream>
#include "CUDA_basic.h"
#include "sharedDefination.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

using namespace std;

//#include "shape.h"
//#include "AAM_common.h"
//#include "texture.h"
//#include "imgAli_common.h"
//#include "AAM_train.h"
//#include <omp.h>
//#include <mmsystem.h>
//#include "opencv2/gpu/gpu.hpp"
//#include "MM.h"

//using namespace cv::gpu;





struct AAM_Search_RealGlobal_CUDA
{
	AAM_Search_RealGlobal_CUDA(int _blockNum=10)
	{

	/*	vec_s_vec.resize(MAX_COUNT_NUM);
		vec_t_vec.resize(MAX_COUNT_NUM);
		vec_s_weight.resize(MAX_COUNT_NUM);
		vec_t_weight.resize(MAX_COUNT_NUM);
		cu_s_vec=raw_pointer_cast(&vec_s_vec[0]);
		cu_t_vec=raw_pointer_cast(&vec_t_vec[0]);
		cu_t_weight=raw_pointer_cast(&vec_t_weight[0]);
		cu_s_weight=raw_pointer_cast(&vec_s_weight[0]);

		vec_s_mean.resize(MAX_COUNT_NUM);
		vec_t_mean.resize(MAX_PIXEL_NUM);
		cu_s_mean=raw_pointer_cast(&cu_s_mean[0]);
		cu_t_mean=raw_pointer_cast(&cu_t_mean[0]);

		vec_currentShape.resize(MAX_COUNT_NUM);
		vec_currentTexture.resize(MAX_PIXEL_NUM);
		cu_currentShape=raw_pointer_cast(&vec_currentShape[0]);
		cu_currentTexture=raw_pointer_cast(&vec_currentTexture[0]);*/
		cudaSetDeviceFlags(cudaDeviceMapHost);
		CUBLAS_CALL(cublasCreate(&blas_handle_));
		CULA_CALL(culaInitialize());
		showSingleStep=false;
		isInputBinded=false;
		
		currentFrameID=0;
		blockNum=_blockNum;
	}

	//AAM_Search_RealGlobal_CUDA(int t)
	//{

	///*	vec_s_vec.resize(MAX_COUNT_NUM);
	//	vec_t_vec.resize(MAX_COUNT_NUM);
	//	vec_s_weight.resize(MAX_COUNT_NUM);
	//	vec_t_weight.resize(MAX_COUNT_NUM);
	//	cu_s_vec=raw_pointer_cast(&vec_s_vec[0]);
	//	cu_t_vec=raw_pointer_cast(&vec_t_vec[0]);
	//	cu_t_weight=raw_pointer_cast(&vec_t_weight[0]);
	//	cu_s_weight=raw_pointer_cast(&vec_s_weight[0]);

	//	vec_s_mean.resize(MAX_COUNT_NUM);
	//	vec_t_mean.resize(MAX_PIXEL_NUM);
	//	cu_s_mean=raw_pointer_cast(&cu_s_mean[0]);
	//	cu_t_mean=raw_pointer_cast(&cu_t_mean[0]);

	//	vec_currentShape.resize(MAX_COUNT_NUM);
	//	vec_currentTexture.resize(MAX_PIXEL_NUM);
	//	cu_currentShape=raw_pointer_cast(&vec_currentShape[0]);
	//	cu_currentTexture=raw_pointer_cast(&vec_currentTexture[0]);*/
	//	//CUBLAS_CALL(cublasCreate(&blas_handle_));
	//	//CULA_CALL(culaInitialize());
	//	cudaSetDeviceFlags(cudaDeviceMapHost);

	//	showSingleStep=false;
	//	isInputBinded=false;
	//}
	float *cu_s_vec,*cu_t_vec,*cu_s_weight,*cu_t_weight;
	float *cu_s_vec_2D;
	//device_vector<float> vec_s_vec,vec_t_vec,vec_s_weight,vec_t_weight;

	float *cu_s_mean,*cu_t_mean;
	float *cu_sMean_T_mean;
	//device_vector<float> vec_s_mean,vec_t_mean;

	thrust::device_vector<float> vec_curShape_curTemplate,vec_curTexture,vec_errorImage;
	float *cu_currentShape,*cu_currentTemplate;
	float *cu_currentLocalShape;
	float *cu_curLocalShape_curTemplate;
	//device_vector<float> vec_currentShape,vec_currentTemplate;
	cublasHandle_t blas_handle_;

	float *cu_warp_tabel,*cu_triangle_indexTabel;
	float *cu_warp_tabel_transpose,*cu_triangle_indexTabel_transpose;
	//device_vector<float> vec_warp_tabel,vec_triangle_indexTabel;

	float cu_theta,cu_scale,cu_translationX,cu_translationY;

	thrust::device_vector<float> Vec_lastShape,Vec_currentShape;

	int s_dim,t_dim;
	int ptsNum,pix_num;

	int fullPix_num;

	int t_width,t_height;

	float *cu_shapeJacobians,*cu_shapeJacobians_transpose;
	float *cu_Jacobians,*cu_RT_Jacobian_transpose;
	float *cu_Jacobian_transpose;

	float *cu_FullJacobians;

	float *cu_fowardIndexTabel;
	//device_vector<float> vec_shapeJacobians;


	//parameters on the run

	float *cu_parameters,*cu_deltaParameters;
	float *cu_inputImg;

	float *cu_allOnesForImg;

	float *cu_currentTexture,*cu_fullCurrentTexture;
	float *cu_MaskTabel;

	float *cu_errorImage;
	float *cu_fullErrorImage;

	float *cu_gradientX,*cu_gradientY;
	float *cu_WarpedGradientX,*cu_WarpedGradientY;

	float *cu_warpedGx_warpedGy;

	float *cu_Hessian;
	float *cu_RTHessian;


	float *cu_lastShape;

	float *host_Hessian,*host_inv_Hessian;
	float *host_b,*cu_deltaB;

	float *cu_inv_Hessian;


	bool showSingleStep;

	bool isInputBinded;

	float *cu_JConv;
	//device_vector<float> vec_parameters;
	
	int *ipiv;

	int currentFrameID;
	float *cu_blockTextureData,*host_blockTextureData;
	int blockNum;
	//Adp_PCA_float *exp;
	bool isAdptive;

	float *newMeanAndTVec;
};




#endif
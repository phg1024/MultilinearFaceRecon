#include <string>
#include <fstream>
#include "CUDA_basic.h"
#include <math.h>
//#include <cutil.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <helper_math.h>

//#include "MM.h"
using namespace std;


#include "AAM_RealGlobal_CUDA.h"


AAM_Search_RealGlobal_CUDA AAM_DataEngine;

int MCN=MAX_COUNT_NUM;
int MAXIMUMPOINTDIM=MAX_PIXEL_NUM;
int MPD=MAX_POINT_DIM;

texture<float,2> TEX_inputImg;
texture<float,3> TEX_triangleIndex;
texture<float,3> TEX_warpTabel;
cudaChannelFormatDesc desc_AAM = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc desc_3D = cudaCreateChannelDesc<float>();
//texture<float,2> outputImg;

//__constant__ float cu_currentParameters[500];
//__constant__ cu_fowardTabel[125000];
void Mat2GPUVEC(double **matrix,int rows,int height,float *cu_vec)
{
	float *host_vec=new float[MCN*sizeof(float)];
	
	int i,j;
	for(i=0;i<rows;i++)
	{
		for(j=0;j<height;j++)
		{
			host_vec[i*height+j]=matrix[i][j];
		}
	}
	
	CUDA_CALL(cudaMemcpy(cu_vec,host_vec,MCN*sizeof(float),cudaMemcpyHostToDevice));
	delete []host_vec;
}


//void GPUVEC2CPUVEC(float *cu_vec,)

void setData(double **m_s_vec,int s_rows,int s_cols,double **m_t_vec, int m_rows,int m_cols)
{
	Mat2GPUVEC(m_s_vec,s_rows,s_cols,AAM_DataEngine.cu_s_vec);
	Mat2GPUVEC(m_t_vec,m_rows,m_cols,AAM_DataEngine.cu_t_vec);
}

__global__ void vectorSubVal(float *vec,float val,int N)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<N)
	{
		vec[offset]-=val;
		//vec[offset]=floor(vec[offset]);
	}
}

void cu_getCurrentTexture(float *t_mean,float *t_weight,float *t_vec,int t_dim,int pix_num)
{
	float alpha = 1.0f;
	float beta=1.0f;
	int m,n,k,lda,ldb,ldc;
	m=1;
	n=pix_num;
	k=t_dim;

	lda=1;
	ldb=pix_num;
	ldc=1;
	
	CUBLAS_CALL(cublasSgemm_v2(AAM_DataEngine.blas_handle_,CUBLAS_OP_N,CUBLAS_OP_T,m,n,k,&alpha,t_weight,lda,t_vec,ldb,&beta,t_mean,ldc));	
}

__global__ void getFullShape(float *currentLocalShape, int ptsNum,int totalDim,float *parameters,float *finalShape)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<ptsNum)
	{
		float theta=parameters[totalDim];
		float scale=parameters[totalDim+1];
		float translationX=parameters[totalDim+2];
		float translationY=parameters[totalDim+3];
		finalShape[offset]=scale*(cos(theta)*currentLocalShape[offset]-sin(theta)*currentLocalShape[offset+ptsNum])+translationX;
		finalShape[offset+ptsNum]=scale*(sin(theta)*currentLocalShape[offset]+cos(theta)*currentLocalShape[offset+ptsNum])+translationY;
	}
}
void cu_getCurrentShape(float *s_mean,float *s_weight,float *s_vec,int s_dim,int t_dim,int ptsNum,float *parameters,float *finalShape)
{
	float alpha = 1.0f;
	float beta=1.0f;
	int m,n,k,lda,ldb,ldc;
	m=1;
	n=ptsNum*2;
	k=s_dim;

	lda=1;
	ldb=ptsNum*2;
	ldc=1;

	CUBLAS_CALL(cublasSgemm_v2(AAM_DataEngine.blas_handle_,CUBLAS_OP_N,CUBLAS_OP_T,m,n,k,&alpha,s_weight,lda,s_vec,ldb,&beta,s_mean,ldc));

	//now smean is the local shape
	int totaDim=s_dim+t_dim;
	getFullShape<<<(ptsNum+128)/128,128>>>(s_mean,ptsNum,totaDim,parameters,finalShape);

}

void test_getTexture()
{
	float *A=new float[MCN*sizeof(float)];
	float *B=new float[MCN*sizeof(float)];
	float *C=new float[MCN*sizeof(float)];

	int i,j;
	for (i=0;i<3;i++)
	{
		A[i]=i+1;
		for (j=0;j<4;j++)
		{
			B[i*4+j]=i*4+j+1;
		}
	}
	for (i=0;i<4;i++)
	{
		C[i]=i+1;
	}
	cout<<"in\n";
	float *cu_A,*cu_B,*cu_C;
	CUDA_CALL(cudaMalloc((void **)&cu_A,MCN*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_B,MCN*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_C,MCN*sizeof(float)));
	CUDA_CALL(cudaMemcpy(cu_A,A,MCN*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cu_B,B,MCN*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cu_C,C,MCN*sizeof(float),cudaMemcpyHostToDevice));
	cout<<"begin\n";

	cu_getCurrentTexture(cu_C,cu_A,cu_B,3,4);
	float *host_result_vec=new float[MCN*sizeof(float)];
	CUDA_CALL(cudaMemcpy(host_result_vec,cu_C,MCN*sizeof(float),cudaMemcpyDeviceToHost));

	for (i=0;i<4;i++)
	{
		cout<<host_result_vec[i]<<" ";
	}
}

//__global__ void cu_PAWarping(float *warp_tabel,float *triangle_indexTabel,float *pts_pos_x,float *pts_pos_y,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
//{
//	/*int x1=blockIdx.x;
//	int y1=blockIdx.y;*/
//	int offset=threadIdx.x+blockIdx.x*blockDim.x;
//
//
//	if (offset<outputWidth*outputHeight)
//	{
//		int ind1=triangle_indexTabel[3*offset];
//		if (ind1!=-1)
//		{
//			int ind2=triangle_indexTabel[3*offset+1];
//			int ind3=triangle_indexTabel[3*offset+2];
//			float x=0;
//			float y=0;
//			x=warp_tabel[offset*3+0]*pts_pos_x[ind1]+
//				warp_tabel[offset*3+1]*pts_pos_x[ind2]+
//				warp_tabel[offset*3+2]*pts_pos_x[ind3];
//			y=warp_tabel[offset*3+0]*pts_pos_y[ind1]+
//				warp_tabel[offset*3+1]*pts_pos_y[ind2]+
//				warp_tabel[offset*3+2]*pts_pos_y[ind3]; 
//
//			int intX,intY;
//			float ratioX,ratioY;
//			float tpx1,tpx2;
//
//			intX=(int)x;
//			intY=(int)y;
//			ratioX=(x-intX);
//			ratioY=y-intY;
//			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
//			m_img.at<uchar>(intY,intX+1);
//			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
//			m_img.at<uchar>(intY+1,intX+1);*/
//			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
//				inputImg[(intY*width+(intX+1))];
//			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
//				inputImg[((intY+1)*width+(intX+1))];
//
//			outputImg[offset]=(1-ratioY)*tpx1+ratioY*tpx2;
//			//outputImg[offset]=inputImg[offset];
//		}
//		else
//		{
//			outputImg[offset]=0;
//		}
//
//	}
//}

__global__ void cu_PAWarping(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
				inputImg[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
				inputImg[((intY+1)*width+(intX+1))];

			outputImg[offset]=(int)((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on
			//outputImg[offset]=inputImg[offset];
		}
		else
		{
			outputImg[offset]=0;
		}

	}
}


__global__ void cu_PAWarping_TEX(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			float intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;


			tpx1=(1-ratioX)*tex2D(TEX_inputImg,intX,intY)+ratioX*
				tex2D(TEX_inputImg,intX+1,intY);
			tpx2=(1-ratioX)*tex2D(TEX_inputImg,intX,intY+1)+ratioX*
				tex2D(TEX_inputImg,intX+1,intY+1);

			outputImg[offset]=(int)((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on
			//outputImg[offset]=tex2D(TEX_inputImg,x,y);
			//outputImg[offset]=tex2D(TEX_inputImg,intY,intX);
		}
		else
		{
			outputImg[offset]=0;
		}

	}
}

__global__ void cu_PAWarping_float(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
				inputImg[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
				inputImg[((intY+1)*width+(intX+1))];

			outputImg[offset]=((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on
			//outputImg[offset]=inputImg[offset];
		}
		else
		{
			outputImg[offset]=0;
		}

	}
}

__global__ void cu_PAWarping_Gradient(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImgX,float *inputImgY,int width,int height,float *outputImgX,float *outputImgY,int outputWidth,int outputHeight,double scale)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImgX[(intY*width+intX)]+ratioX*
				inputImgX[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImgX[((intY+1)*width+intX)]+ratioX*
				inputImgX[((intY+1)*width+(intX+1))];

			outputImgX[offset]=(((1-ratioY)*tpx1+ratioY*tpx2))*scale;///////int can be deleted later on
			
			tpx1=(1-ratioX)*inputImgY[(intY*width+intX)]+ratioX*
				inputImgY[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImgY[((intY+1)*width+intX)]+ratioX*
				inputImgY[((intY+1)*width+(intX+1))];

			outputImgY[offset]=(((1-ratioY)*tpx1+ratioY*tpx2))*scale;///////int can be deleted later on
			//outputImg[offset]=inputImg[offset];
		}
		else
		{
			outputImgX[offset]=0;
			outputImgY[offset]=0;
		}

	}
}

__global__ void VectorSubtraction(float *vec1,float *vec2,float *vec3,int N)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<N)
	{
		vec3[offset]=vec1[offset]-vec2[offset];
	}
}

__global__ void getJacobians_Preprocess(float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,int pixelNum,float *warp_tabel,
										float *t_vec,float *fowardIndex)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			int allDim=s_dim+t_dim+4;
			int i=0;
			for (i=s_dim;i<s_dim+t_dim;i++)
			{
				Jacobians[pixelID*allDim+i]=t_vec[(i-s_dim)*pixelNum+pixelID];
			}
		}
		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}

		//}
	}
}

__global__ void getJacobians_fullParral(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum)
{
		int pixel_id=blockIdx.x*blockDim.x+threadIdx.x;
	int dim_id=blockIdx.y*blockDim.y+threadIdx.y;




	if (pixel_id<t_width*t_height&&dim_id<s_dim+t_dim+4)
	{
		int pixelID=fowardIndex[pixel_id];
		if (pixelID!=-1)
		{
			int totalDim=s_dim+t_dim;
			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];

			
			//float s_theta,s_scale,s_translationX,s_translationY;
			//int s_triangle_indexTabel[3];
			//float s_warpedTabel[3];
			//
			//float s_shapeJacobians[2]; //see shapeJacobians first
			//if(dim_id<s_dim)
			//{
			//	s_shapeJacobians[0]=shapeJacobians[2*s_dim*pixel_id+dim_id];
			//	s_shapeJacobians[1]=shapeJacobians[2*s_dim*pixel_id+dim_id+s_dim];
			//}

			
		/*	s_triangle_indexTabel[0]=triangle_indexTabel[3*pixel_id];
			s_triangle_indexTabel[1]=triangle_indexTabel[3*pixel_id+1];
			s_triangle_indexTabel[2]=triangle_indexTabel[3*pixel_id+2];
			s_warpedTabel[0]=warp_tabel[3*pixel_id];
			s_warpedTabel[1]=warp_tabel[3*pixel_id+1];
			s_warpedTabel[2]=warp_tabel[3*pixel_id+2];*/
		/*	float s_gradientX=gradientX[pixel_id];
			float s_gradientY=gradientY[pixel_id];*/	


		
			

			
			float costheta=cos(theta);float sintheta=sin(theta);

			float gx=gradientX[pixel_id];float gy=gradientY[pixel_id];

		

		
			float cgradient[2];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);
			

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			if(dim_id<s_dim)
			{
			//	Jacobians[cTotalID+dim_id]=scale*(cgradient[0]*shapeJacobians[2*s_dim*pixel_id+dim_id]+
			//		cgradient[1]*shapeJacobians[2*s_dim*pixel_id+dim_id+s_dim]);
			
				Jacobians[cTotalID+dim_id]=scale*(cgradient[0]*shapeJacobians[2*s_dim*pixel_id+dim_id]+
							cgradient[1]*shapeJacobians[2*s_dim*pixel_id+dim_id+s_dim]);
			}
			else if (dim_id==s_dim+t_dim)
			{
				float cgradient_theta[2];
				cgradient_theta[0]=-(gx*-sintheta+gy*costheta);
				cgradient_theta[1]=-(gx*-costheta+gy*-sintheta);
				int tInd0=triangle_indexTabel[3*pixel_id];
				int tInd1=triangle_indexTabel[3*pixel_id+1];
				int tInd2=triangle_indexTabel[3*pixel_id+2];

				float alpha=warp_tabel[3*pixel_id];
				float beta=warp_tabel[3*pixel_id+1];
				float gamma=warp_tabel[3*pixel_id+2];

				float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
					gamma*currentLocalShape[tInd2];
				float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
					gamma*currentLocalShape[tInd2+ptsNum];
				Jacobians[cTotalID+dim_id]=scale*(cgradient_theta[0]*sumtx+cgradient_theta[1]*sumty);
				//Jacobians[cTotalID+dim_id]=scale*cgradient_theta[1]*sumty;//the last 4 parameters are wrong. But why?
			}
			else if(dim_id==s_dim+t_dim+1)
			{
				int tInd0=triangle_indexTabel[3*pixel_id];
				int tInd1=triangle_indexTabel[3*pixel_id+1];
				int tInd2=triangle_indexTabel[3*pixel_id+2];

				float alpha=warp_tabel[3*pixel_id];
				float beta=warp_tabel[3*pixel_id+1];
				float gamma=warp_tabel[3*pixel_id+2];

				float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
					gamma*currentLocalShape[tInd2];
				float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
					gamma*currentLocalShape[tInd2+ptsNum];
				Jacobians[cTotalID+dim_id]=cgradient[0]*sumtx+cgradient[1]*sumty;
				//Jacobians[cTotalID+dim_id]=cgradient[0]*sumtx+cgradient[1]*sumty;
			}
			else if (dim_id==s_dim+t_dim+2)
			{
				Jacobians[cTotalID+dim_id]=-gx;
			}
			else if (dim_id==s_dim+t_dim+3)
			{
				Jacobians[cTotalID+dim_id]=-gy;
			}
		}	
	}
}


const int blockDIM=16;
const int blockDIMX=blockDIM;
__global__ void getJacobians_shared(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum)
{
	int pixel_id=blockIdx.x*blockDim.x+threadIdx.x;
	int dim_id=blockIdx.y*blockDim.y+threadIdx.y;

	int x=threadIdx.x;
	int y=threadIdx.y;

	int totalDim=s_dim+t_dim;
	__shared__ 	float theta;
	__shared__  float scale;
	__shared__  float translationX;
	__shared__  float translationY;

	__shared__ float gx[blockDIMX+1];
	__shared__ float gy[blockDIMX+1];

	__shared__ int tInd0[blockDIMX+1];
	__shared__ int tInd1[blockDIMX+1];
	__shared__ int tInd2[blockDIMX+1];

	__shared__ float alpha[blockDIMX+1];
	__shared__ float beta[blockDIMX+1];
	__shared__ float gamma[blockDIMX+1];

	__shared__ float s_shapeJacobians[blockDIMX][blockDIM*2+1];

	if (pixel_id<t_width*t_height&&dim_id<s_dim+t_dim+4)
	{

		int pixelID=fowardIndex[pixel_id];
		if (pixelID!=-1)
		{
			s_shapeJacobians[x][y]=shapeJacobians[2*s_dim*pixel_id+dim_id];
			s_shapeJacobians[x][y+blockDIM]=shapeJacobians[2*s_dim*pixel_id+dim_id+s_dim];
		}
		if (x==0&&y==0)
		{
			theta=parameters[totalDim];
			scale=parameters[totalDim+1];
			translationX=parameters[totalDim+2];
			translationY=parameters[totalDim+3];
		}
		
		if (y==0)
		{
			gx[x]=gradientX[pixel_id];
			gy[x]=gradientY[pixel_id];

			tInd0[x]=triangle_indexTabel[3*pixel_id];
			tInd1[x]=triangle_indexTabel[3*pixel_id+1];
			tInd2[x]=triangle_indexTabel[3*pixel_id+2];

			alpha[x]=warp_tabel[3*pixel_id];
			beta[x]=warp_tabel[3*pixel_id+1];
			gamma[x]=warp_tabel[3*pixel_id+2];
		}
		
	}
	__syncthreads();

	if (pixel_id<t_width*t_height&&dim_id<s_dim+t_dim+4)
	{
		int pixelID=fowardIndex[pixel_id];
		if (pixelID!=-1)
		{
			
		
			
			float costheta=cos(theta);float sintheta=sin(theta);

			

		

		
			float cgradient[2];
			cgradient[0]=-(gx[x]*costheta+gy[x]*sintheta);
			cgradient[1]=-(gx[x]*(-sintheta)+gy[x]*costheta);
			

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			if(dim_id<s_dim)
			{
			//	Jacobians[cTotalID+dim_id]=scale*(cgradient[0]*shapeJacobians[2*s_dim*pixel_id+dim_id]+
			//		cgradient[1]*shapeJacobians[2*s_dim*pixel_id+dim_id+s_dim]);
			
				Jacobians[cTotalID+dim_id]=scale*(cgradient[0]*s_shapeJacobians[x][y]+
							cgradient[1]*s_shapeJacobians[x][blockDIM+y]);
			//	Jacobians[cTotalID+dim_id]=dim_id;
			}
			else if (dim_id==s_dim+t_dim)
			{
				float cgradient_theta[2];
				cgradient_theta[0]=-(gx[x]*-sintheta+gy[x]*costheta);
				cgradient_theta[1]=-(gx[x]*-costheta+gy[x]*-sintheta);
			/*	int tInd0=triangle_indexTabel[3*pixel_id];
				int tInd1=triangle_indexTabel[3*pixel_id+1];
				int tInd2=triangle_indexTabel[3*pixel_id+2];

				float alpha=warp_tabel[3*pixel_id];
				float beta=warp_tabel[3*pixel_id+1];
				float gamma=warp_tabel[3*pixel_id+2];*/

				float sumtx=alpha[x]*currentLocalShape[tInd0[x]]+beta[x]*currentLocalShape[tInd1[x]]+
					gamma[x]*currentLocalShape[tInd2[x]];
				float sumty=alpha[x]*currentLocalShape[tInd0[x]+ptsNum]+beta[x]*currentLocalShape[tInd1[x]+ptsNum]+
					gamma[x]*currentLocalShape[tInd2[x]+ptsNum];
				Jacobians[cTotalID+dim_id]=scale*(cgradient_theta[0]*sumtx+cgradient_theta[1]*sumty);
				//Jacobians[cTotalID+dim_id]=scale*cgradient_theta[1]*sumty;//the last 4 parameters are wrong. But why?
			}
			else if(dim_id==s_dim+t_dim+1)
			{
				/*int tInd0=triangle_indexTabel[3*pixel_id];
				int tInd1=triangle_indexTabel[3*pixel_id+1];
				int tInd2=triangle_indexTabel[3*pixel_id+2];

				float alpha=warp_tabel[3*pixel_id];
				float beta=warp_tabel[3*pixel_id+1];
				float gamma=warp_tabel[3*pixel_id+2];*/

				float sumtx=alpha[x]*currentLocalShape[tInd0[x]]+beta[x]*currentLocalShape[tInd1[x]]+
					gamma[x]*currentLocalShape[tInd2[x]];
				float sumty=alpha[x]*currentLocalShape[tInd0[x]+ptsNum]+beta[x]*currentLocalShape[tInd1[x]+ptsNum]+
					gamma[x]*currentLocalShape[tInd2[x]+ptsNum];
				Jacobians[cTotalID+dim_id]=cgradient[0]*sumtx+cgradient[1]*sumty;
				//Jacobians[cTotalID+dim_id]=cgradient[0]*sumtx+cgradient[1]*sumty;
			}
			else if (dim_id==s_dim+t_dim+2)
			{
				Jacobians[cTotalID+dim_id]=-gx[x];
			}
			else if (dim_id==s_dim+t_dim+3)
			{
				Jacobians[cTotalID+dim_id]=-gy[x];
			}
		}	
	}

//	__syncthreads();
//	return;

	//int pixel_id=x_index;
	//int dim_id=y_index;
	
}

__global__ void getJacobians_TEX(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			int totalDim=s_dim+t_dim;

			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];
		
		/*	int tInd0=triangle_indexTabel[3*offset];
			int tInd1=triangle_indexTabel[3*offset+1];
			int tInd2=triangle_indexTabel[3*offset+2];

			
			float alpha=warp_tabel[offset*3+0];
			float beta=warp_tabel[offset*3+1];
			float gamma=warp_tabel[offset*3+2];*/

			int x=offset%t_width;
			int y=offset/t_width;
			int tInd0=tex3D(TEX_triangleIndex,x,y,0);
			int tInd1=tex3D(TEX_triangleIndex,x,y,1);
			int tInd2=tex3D(TEX_triangleIndex,x,y,2);


			float alpha=tex3D(TEX_warpTabel,x,y,0);
			float beta=tex3D(TEX_warpTabel,x,y,1);
			float gamma=tex3D(TEX_warpTabel,x,y,2);


			float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
				gamma*currentLocalShape[tInd2];
			float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
				gamma*currentLocalShape[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);

			
			
			Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians[cTotalID+s_dim+t_dim+3]=-gy;

			//smooth weight
		}
	}
}

__global__ void getJacobians(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			int totalDim=s_dim+t_dim;

			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];
		
			int tInd0=triangle_indexTabel[3*offset];
			int tInd1=triangle_indexTabel[3*offset+1];
			int tInd2=triangle_indexTabel[3*offset+2];


			float alpha=warp_tabel[offset*3+0];
			float beta=warp_tabel[offset*3+1];
			float gamma=warp_tabel[offset*3+2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
				gamma*currentLocalShape[tInd2];
			float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
				gamma*currentLocalShape[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);

			
			
			Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians[cTotalID+s_dim+t_dim+3]=-gy;

			//smooth weight
		}
		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}
		//	
		//}
	}
}

extern "C" void test_PAWarping(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	//set data
	float *cu_warpTabel,*cu_triIndex,*cu_ptsPos,*cu_inputImg,*cu_OutputImg;
	
	CUDA_CALL(cudaMalloc((void **)&cu_warpTabel,MAXIMUMPOINTDIM*3*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_triIndex,MAXIMUMPOINTDIM*3*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_ptsPos,MCN*sizeof(float)));
	//CUDA_CALL(cudaMalloc((void **)&cu_ptsPosY,MCN*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_inputImg,MAXIMUMPOINTDIM*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&cu_OutputImg,MAXIMUMPOINTDIM*sizeof(float)));

	CUDA_CALL(cudaMemcpy(cu_warpTabel,warp_tabel,MAXIMUMPOINTDIM*3*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cu_triIndex,triangle_indexTabel,MAXIMUMPOINTDIM*3*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cu_ptsPos,pts_pos,MCN*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(cu_ptsPosY,pts_pos_y,MCN*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cu_inputImg,inputImg,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));
	
	//call function
	//cu_PAWarping<<<MAXIMUMPOINTDIM/256,256>>>(cu_warpTabel,cu_triIndex,cu_ptsPos,ptsNum,cu_inputImg,width,height,cu_OutputImg,outputWidth,outputHeight);
	
	/*TEX_inputImg.filterMode=cudaFilterModeLinear;
	TEX_inputImg.addressMode[0]=cudaAddressModeClamp;
	TEX_inputImg.addressMode[1]=cudaAddressModeClamp;
	TEX_inputImg.normalized=true;*/
	//TEX_inputImg.filterMode=cudaFilterModePoint;
	CUDA_CALL(cudaBindTexture2D( NULL, TEX_inputImg,
		cu_inputImg,
		desc_AAM,  width,height,
		sizeof(float) * width));//the last one should be max(data->t_width,data->t_height) ) 
	cu_PAWarping_TEX<<<MAXIMUMPOINTDIM/256,256>>>(cu_warpTabel,cu_triIndex,cu_ptsPos,ptsNum,cu_inputImg,width,height,cu_OutputImg,outputWidth,outputHeight);
	
	//copy back result
	CUDA_CALL(cudaMemcpy(outputImg,cu_OutputImg,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyDeviceToHost));

	//ofstream out("texture.txt",ios::out);
	//for (int i=0;i<outputHeight;i++)
	//{
	//	for (int j=0;j<outputWidth;j++)
	//	{
	//	//	if (outputImg[i*outputWidth+j]!=0)
	//		{
	//			out<<outputImg[i*outputWidth+j]<<" ";
	//		}
	//	}
	//	out<<endl;
	//}

	cudaFree(cu_warpTabel);
	cudaFree(cu_triIndex);
	cudaFree(cu_ptsPos);
	//cudaFree(cu_ptsPosY);
	cudaFree(cu_inputImg);
	cudaFree(cu_OutputImg);

}

void MatrixMVector(float *Matrix,int rows,int cols,float *_Vector,float *resultVec,AAM_Search_RealGlobal_CUDA *data)
{
	float alpha=1.0;
	float beta=0;
	CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,rows,cols,&alpha,Matrix,rows,_Vector,1,&beta,
		resultVec,1));
}


extern "C" void setData_Preprocess(float *s_vec,float *t_vec,float *s_mean,float *t_mean,
					  float *warpTabel,float *triangle_indexTabel,int s_dim,int t_dim,int ptsNum,int pix_num,
					  int t_width,int t_height,float *shapeJacobians,float *maskTabel,float *fowardIndex,bool showSingleStep)
{
	AAM_Search_RealGlobal_CUDA *data=&AAM_DataEngine;

	CUDA_CALL( cudaMalloc(&data->cu_s_vec, MCN * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_s_vec,s_vec,MCN*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL( cudaMalloc(&data->cu_t_vec, t_dim*MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_t_vec,t_vec,t_dim*MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));



	CUDA_CALL( cudaMalloc(&data->cu_t_mean, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_t_mean,t_mean,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL( cudaMalloc(&data->cu_warp_tabel, MAXIMUMPOINTDIM*3 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_warp_tabel,warpTabel,MAXIMUMPOINTDIM*3*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL( cudaMalloc(&data->cu_triangle_indexTabel, MAXIMUMPOINTDIM * 3 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_triangle_indexTabel,triangle_indexTabel,MAXIMUMPOINTDIM*3*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL( cudaMalloc(&data->cu_shapeJacobians, MAXIMUMPOINTDIM*s_dim*2 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_shapeJacobians,shapeJacobians,MAXIMUMPOINTDIM*s_dim*2*sizeof(float),cudaMemcpyHostToDevice));


	//s_weight,t_weight
	CUDA_CALL( cudaMalloc(&data->cu_s_weight, MPD * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_t_weight, MPD * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_currentLocalShape, MPD * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_currentShape, MPD* sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_currentTemplate, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_errorImage, pix_num * sizeof(float)) );




	CUDA_CALL( cudaMalloc(&data->cu_s_mean, MPD * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_s_mean,s_mean,MPD*sizeof(float),cudaMemcpyHostToDevice));

	CUDA_CALL( cudaMalloc(&data->cu_parameters, MPD * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_deltaParameters, MPD * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_inputImg, MAXIMUMPOINTDIM * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&data->cu_currentTexture, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_fullCurrentTexture, MAXIMUMPOINTDIM * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&data->cu_MaskTabel, t_width*t_height * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_MaskTabel,maskTabel,t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));

	CUDA_CALL( cudaMalloc(&data->cu_gradientX, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_gradientY, MAXIMUMPOINTDIM * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&data->cu_WarpedGradientX, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_WarpedGradientY, MAXIMUMPOINTDIM * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_Hessian, (s_dim+t_dim+4)*(s_dim+t_dim+4) * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&data->cu_Jacobians, t_width*t_height*(s_dim+t_dim+4) * sizeof(float)) );
	
	
	CUDA_CALL( cudaMalloc(&data->cu_fullErrorImage, t_width*t_height* sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_lastShape, MPD* sizeof(float)) );
	//cout<<t_width*t_height*(s_dim+t_dim+4)<<endl;


	//CUDA_CALL( cudaMalloc(&data->cu_fowardIndexTabel, t_width*t_height * sizeof(float)) );
	//CUDA_CALL(cudaMemcpy(data->cu_fowardIndexTabel,fowardIndex,t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));

	data->cu_fowardIndexTabel=data->cu_MaskTabel;

	//float *cJacobians=new float[(s_dim+t_dim+4)*t_width*t_height];
	//CUDA_CALL(cudaMemcpy(cJacobians, data->cu_Jacobians, (s_dim+t_dim+4)*t_width*t_height*sizeof(float), cudaMemcpyDeviceToHost ));
	//for (int i=0;i<50;i++)
	//{
	//	cout<<cJacobians[i]<<" ";
	//}
	//cout<<endl;

	float *allones=new float[MAXIMUMPOINTDIM];
	for (int i=0;i<pix_num;i++)
	{
		allones[i]=1;
	}
	CUDA_CALL(cudaMalloc(&data->cu_allOnesForImg,MAXIMUMPOINTDIM*sizeof(float)));
	CUDA_CALL(cudaMemcpy(data->cu_allOnesForImg,allones,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));


	data->s_dim=s_dim;
	data->t_dim=t_dim;
	data->ptsNum=ptsNum;
	data->pix_num=pix_num;
	data->t_width=t_width;
	data->t_height=t_height;
	
	data->fullPix_num=data->t_width*data->t_height;
	
	delete []allones;

	getJacobians_Preprocess<<<t_width*t_height/128+1,128>>>(data->cu_Jacobians,t_width,t_height,s_dim,t_dim,data->pix_num,data->cu_warp_tabel,data->cu_t_vec,data->cu_fowardIndexTabel);

	/*data->host_Hessian=new float[(t_dim+s_dim+4)*(t_dim+s_dim+4)];
	data->host_inv_Hessian=new float[(t_dim+s_dim+4)*(t_dim+s_dim+4)];*/

	CUDA_CALL(cudaHostAlloc(&data->host_Hessian,(t_dim+s_dim+4)*(t_dim+s_dim+4)*sizeof(float),cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc(&data->host_inv_Hessian,(t_dim+s_dim+4)*(t_dim+s_dim+4)*sizeof(float),cudaHostAllocMapped));
	cudaHostGetDevicePointer((void **)&data->cu_inv_Hessian, (void *)data->host_inv_Hessian, 0);


	data->showSingleStep=showSingleStep;

	//TEX_inputImg.filterMode=cudaFilterModePoint;
	//
	////set up texture memory for triangleIndex and warptabel
	//unsigned int  size=t_width*t_height*3;
	//cudaArray *cu_triangleIndexArray=0;
	//cudaExtent extent;     
	//extent.width=t_width;     
	//extent.height=t_height;     
	//extent.depth=3; 
	//cudaMalloc3DArray(&cu_triangleIndexArray,&desc_3D,extent);
	//cudaMemcpy3DParms copyParams = {0};   
	//copyParams.srcPtr   = make_cudaPitchedPtr((void*)triangle_indexTabel, t_width*sizeof(float), t_width, t_height);
	//copyParams.dstArray = cu_triangleIndexArray;
	//copyParams.extent   = extent;
	//copyParams.kind     = cudaMemcpyHostToDevice;
	//cudaMemcpy3D(&copyParams);
	//TEX_triangleIndex.filterMode=cudaFilterModePoint;
	//TEX_triangleIndex.normalized=false;
	//TEX_triangleIndex.channelDesc =desc_3D;
	//CUDA_CALL(cudaBindTextureToArray(TEX_triangleIndex,cu_triangleIndexArray,desc_3D));

	//cudaArray *cu_warptabelArray=0;
	//cudaMalloc3DArray(&cu_warptabelArray,&desc_3D,extent);
	//cudaMemcpy3DParms copyParams_warp = {0};   
	//copyParams_warp.srcPtr   = make_cudaPitchedPtr((void*)warpTabel, t_width*sizeof(float), t_width, t_height);
	//copyParams_warp.dstArray = cu_triangleIndexArray;
	//copyParams_warp.extent   = extent;
	//copyParams_warp.kind     = cudaMemcpyHostToDevice;
	//cudaMemcpy3D(&copyParams_warp);
	//TEX_warpTabel.filterMode=cudaFilterModePoint;
	//TEX_warpTabel.normalized=false;
	//TEX_warpTabel.channelDesc =desc_3D;
	//CUDA_CALL(cudaBindTextureToArray(TEX_warpTabel,cu_warptabelArray,desc_3D));
}

//parameters,inputimage
extern "C" void setData_onRun(float *parameters,float *inputImage,float *InputGradientX,float *InputGradientY,int width,int height)
{
	AAM_Search_RealGlobal_CUDA *data=&AAM_DataEngine;

	if (parameters[0]!=-1000000000)
	{
		CUDA_CALL(cudaMemcpy(data->cu_parameters,parameters,MPD*sizeof(float),cudaMemcpyHostToDevice));
	}
	

	//float *cfShape=new float[MPD];
	//CUDA_CALL(cudaMemcpy(cfShape, data->cu_parameters, MPD*sizeof(float), cudaMemcpyDeviceToHost ));
	//cout<<"parameters\n";
	//for (int i=0;i<data->s_dim+data->t_dim+4;i++)
	//{
	//	//cout<<cShape[i]<<" "<<cShape[data->ptsNum+i]<<endl;
	//	cout<<cfShape[i]<<" ";
	//}
	//cout<<endl;

	//CULA_CALL( culaGetOptimalPitch(&pitch, t_dim, pix_num, sizeof(float)) );
	
	CUDA_CALL(cudaMemcpy(data->cu_inputImg,inputImage,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));

	
	CUDA_CALL(cudaMemcpy(data->cu_gradientX,InputGradientX,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_gradientY,InputGradientY,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));


	//warping does not need texture memory to speed up. Jacobian can!
	//if (!data->isInputBinded)
	//{
	//	CUDA_CALL(cudaBindTexture2D( NULL, TEX_inputImg,
	//		data->cu_inputImg,
	//		desc,  width,height,
	//		sizeof(float) * width));//the 
	//	data->isInputBinded=true;
	//}


}

__global__ void setCompactTexture(float *fullTexture,float *compactTexture,float *MaskTabel,int width,int height)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<width*height)
	{
		int ind=MaskTabel[offset];
		if (ind!=-1)
		{
			compactTexture[ind]=fullTexture[offset];
		}
	}
}

__global__ void compactError2Full(float *imageError,float *fullError,float *MaskTabel,int width,int height)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
		int ind=MaskTabel[offset];
		if (ind!=-1)
		{
			fullError[offset]=imageError[ind];
		}
		else
		{
			fullError[offset]=0;
		}
	}
}
/*
what we have:
float *cu_s_vec,*cu_t_vec,*cu_s_weight,*cu_t_weight;
float *cu_s_mean,*cu_t_mean;
float *cu_currentShape,*cu_currentTexture;
float *parameters;
int shape_dim,texture_dim;

float *warp_tabel,*triangle_indexTabel;

[s_weight,t_weight,theta,k,transform_x,transform_y]
*/

__global__ void generateEye(float *E, int size){
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int y = offset / size;
	int x = offset - y * size;
	if(offset > size * size) return;
	if(x == y) E[offset] = 1;
	else E[offset] = 0;
}

int matrixInvert(float *devInput, float *devOutput, int size){
	int *ipiv;
	float *cache;
	int size2 = size * size * sizeof(float);
	int THREADS=128;
	int blocks = (size * size) / THREADS + 1;
	cudaMalloc(&ipiv, size2);
	cudaMalloc(&cache, size2);
	generateEye<<<blocks, THREADS>>>(devOutput, size);
	cudaMemcpy(cache, devInput, size2, cudaMemcpyDeviceToDevice);
	culaInitialize();
	culaDeviceSgesv(size, size, cache, size, ipiv, devOutput, size);
	culaShutdown();
	cudaFree(ipiv);
	cudaFree(cache);
	return 0;
}

extern "C" void iterate_CUDA(int width,int height,double smoothWeight,double AAM_weight,int currentFrame,int startFrame,float *parameters,int **inv_mask)
{
	cout<<"in the cuda AAM\n";
	//return;
	AAM_Search_RealGlobal_CUDA *data=&AAM_DataEngine;

	//float *cu_inputImg=data->cu_inputImg;
	int allDim=data->s_dim+data->t_dim+4;
	int MaxIterNum=60;

	/*if (smoothWeight>0)
	{
		MaxIterNum=35;
	}*/
	
	int incx,incy;
	incx=incy=1;
	//float *cu_currentTexture;
//	float *cu_errorImage;
	float result;
	float textureScale;
	float tmp_scale,tex_scale;

	//float *cu_inputGradient_X,*cu_inputGradient_Y;
	//float *cu_currentGradientX,*cu_currentGradientY;

	//float *Jacobians;//pixelnum*(s_dim+t_dim+4)
	//device_vector<float> d_Hessian(dim*dim);
	//float *Hessian=raw_pointer_cast(&d_Hessian[0]);

	device_vector<int> d_ipiv(allDim);
	int *ipiv=raw_pointer_cast(&d_ipiv[0]);

	//float *lastParameters;
	//malloc Jacobians and Hessian
	//suppose parameters are already initialized

	//calculate the gradient of input image
	//By openCV first
	
	int full_pix_num=data->t_width*data->t_height;


	int times=0;

	float alpha,beta;
	float difference;
		
	//cout<<MPD<<endl;
	//float *cDeltaParameters=new float[MPD];
	//CUDA_CALL(cudaMemcpy(cDeltaParameters, data->cu_deltaParameters,MPD*sizeof(float), cudaMemcpyDeviceToHost ));
	//cout<<"deltaP\n";
	//for (int kk=0;kk<allDim;kk++)
	//{
	//	cout<<cDeltaParameters[kk]<<" ";
	//}
	//cout<<endl;

	float errorSum,lastError;

	
	dim3 grid((data->t_width*data->t_height)/blockDIM+1,(data->s_dim+data->t_dim+4)/blockDIM+1,1);
	dim3 threads(blockDIMX,blockDIM,1);
	//cout<<"lalala\n";

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));
	

	while(1)
	{

		//show current image
		if (data->showSingleStep)
		{
			cout<<"time "<<times<<endl;
			float *parameters=new float[MPD];
			CUDA_CALL(cudaMemcpy(parameters, data->cu_parameters, MPD*sizeof(float), cudaMemcpyDeviceToHost ));
			checkIterationResult(parameters,data->ptsNum,data->s_dim,data->t_dim,false);
			delete []parameters;
		}
		
		//copy parameters to constant memory
		//CUDA_CALL(cudaMemcpyToSymbol(cu_currentParameters,data->cu_parameters,MPD*sizeof(float),0,cudaMemcpyDeviceToDevice));

		//!!need to assign mean value to current shape
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->s_dim,data->cu_parameters,incx,data->cu_s_weight,incy));
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->t_dim,data->cu_parameters+data->s_dim,incx,data->cu_t_weight,incy));
		/*data->cu_theta=data->cu_parameters[data->t_dim+data->s_dim];
		data->cu_scale=data->cu_parameters[data->t_dim+data->s_dim+1];
		data->cu_translationX=data->cu_parameters[data->t_dim+data->s_dim+2];
		data->cu_translationY=data->cu_parameters[data->t_dim+data->s_dim+3];*/
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->ptsNum*2,data->cu_s_mean,incx,data->cu_currentLocalShape,incy));
		cu_getCurrentShape(data->cu_currentLocalShape,data->cu_s_weight,data->cu_s_vec,data->s_dim,data->t_dim,data->ptsNum,data->cu_parameters,data->cu_currentShape);
		//////////////////shape check point//////////////////////

		
	
		/*float *cShape=new float[MPD];
		CUDA_CALL(cudaMemcpy(cShape, data->cu_currentShape, MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		for (int i=0;i<data->ptsNum;i++)
		{
			cout<<cShape[i]<<" "<<cShape[i+data->ptsNum]<<endl;
		}*/
		//break;
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num,data->cu_t_mean,incx,data->cu_currentTemplate,incy));

	
		cu_getCurrentTexture(data->cu_currentTemplate,data->cu_t_weight,data->cu_t_vec,data->t_dim,data->pix_num);
		
		//normalize and devide
		//normalize
		float sum;

		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTemplate,incx,data->cu_allOnesForImg,incy,&sum));
		sum/=data->pix_num;
		vectorSubVal<<<(data->pix_num+128)/128,128>>>(data->cu_currentTemplate,sum,data->pix_num);
		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTemplate,incx,data->cu_t_mean,incy,&sum));
		sum=1.0f/sum;
		CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,data->pix_num,&sum,data->cu_currentTemplate,incx));
		
		//////////////////template check point//////////////////////
		//float *cWeight=new float[MPD];
		//CUDA_CALL(cudaMemcpy(cWeight, data->cu_t_weight, MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"current weight\n";
		//for (int i=0;i<100;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	cout<<cWeight[i]<<" ";
		//}
		//cout<<endl;
		//cout<<data->t_dim<<endl;

		//float *cTemplate=new float[MAXIMUMPOINTDIM];
		//CUDA_CALL(cudaMemcpy(cTemplate, data->cu_currentTemplate, MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"new texture\n";
		//for (int i=0;i<100;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	cout<<cTemplate[i]<<" ";
		//}
		//cout<<endl;

		//float *cEigenT=new float[MAXIMUMPOINTDIM*data->t_dim];
		//CUDA_CALL(cudaMemcpy(cEigenT, data->cu_t_vec, MAXIMUMPOINTDIM*data->t_dim*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"new texture\n";
		//for (int i=0;i<data->t_dim;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	cout<<cEigenT[i]<<" ";
		//}
		//cout<<endl;


		
		//call function
		cu_PAWarping<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_inputImg,width,height,data->cu_fullCurrentTexture,data->t_width,data->t_height);
		//cu_PAWarping_TEX<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_inputImg,width,height,data->cu_fullCurrentTexture,data->t_width,data->t_height);
		

		setCompactTexture<<<(data->t_width*data->t_height+128)/128,128>>>(data->cu_fullCurrentTexture,data->cu_currentTexture,
			data->cu_MaskTabel,data->t_width,data->t_height);

		


		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTexture,incx,data->cu_allOnesForImg,incy,&sum));
		sum/=data->pix_num;
		vectorSubVal<<<(data->pix_num+128)/128,128>>>(data->cu_currentTexture,sum,data->pix_num);

		
		////calculate the texture scale
		//CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTexture,incx,data->cu_currentTexture,incy,&textureScale));
		//tex_scale=1.0f/sqrt(textureScale);

		
		//texture normalize
		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_t_mean,incx,data->cu_currentTexture,incy,&result));
		
		tmp_scale=1.0f/result;
		tex_scale=tmp_scale;
		CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,data->pix_num,&tmp_scale,data->cu_currentTexture,incx));
		
		//////////////////texture check point//////////////////////
		
		//float *cTemplate=new float[MAXIMUMPOINTDIM];
		//CUDA_CALL(cudaMemcpy(cTemplate, data->cu_currentTexture, MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"current input texture\n";
		//for (int i=0;i<100;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	cout<<cTemplate[i]<<" ";
		//}
		//cout<<endl;

	
		


		//calculate image difference
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num,data->cu_currentTemplate,1,data->cu_errorImage,1));
		alpha=-1;
		CUBLAS_CALL(cublasSaxpy_v2(data->blas_handle_,data->pix_num,&alpha,data->cu_currentTexture,1,data->cu_errorImage,1));
		//////////////////erroeImage double checked/////////////////////
		//float *cTemplate=new float[data->pix_num];
		//CUDA_CALL(cudaMemcpy(cTemplate, data->cu_errorImage, data->pix_num*sizeof(float), cudaMemcpyDeviceToHost ));
		//ofstream out_e("E_G.txt",ios::out);
		//for (int i=0;i<data->pix_num;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	out_e<<cTemplate[i]<<" ";
		//}
		//out_e<<endl;
		//out_e.close();

		//cout<<"Error\n";
		//for (int i=0;i<data->pix_num;i++)
		//{
		//	//cnum=inv_mask[i][1]*data->t_width+inv_mask[i][0];
		//	cout<<cTemplate[i]<<" ";
		//}
		//cout<<endl;
		//

		//get whole error
		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_errorImage,incx,data->cu_errorImage,incy,&errorSum));
		//break;
		//////////////////////checked//////////////////////////
		//VectorSubtraction<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_currentTemplate,cu_currentTexture,cu_errorImage);
		
		//get SD image, which is J'*E
		cu_PAWarping_float<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_gradientX,width,height,data->cu_WarpedGradientX,data->t_width,data->t_height);
		cu_PAWarping_float<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_gradientY,width,height,data->cu_WarpedGradientY,data->t_width,data->t_height);		
		CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,full_pix_num,&tex_scale,data->cu_WarpedGradientX,incx));
		CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,full_pix_num,&tex_scale,data->cu_WarpedGradientY,incx));
		//break;

		//cu_PAWarping_Gradient<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_gradientX,data->cu_gradientY,width,height,data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->t_width,data->t_height,tex_scale);

		//////////////////////gradient warping double checked/////////////////////////////////

	/*	float *mwgx=new float[MAXIMUMPOINTDIM];
		CUDA_CALL(cudaMemcpy(mwgx, data->cu_WarpedGradientX,MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_G("gradientX_GPU.txt",ios::out);

		for (int i=0;i<data->t_height;i++)
		{
			for (int j=0;j<data->t_width;j++)
			{
				out_G<<mwgx[i*data->t_width+j]<<" ";
			}
			out_G<<endl;
		}
		out_G.close();*/

		//float *mgx=new float[MAXIMUMPOINTDIM];
		//float *mgy=new float[MAXIMUMPOINTDIM];
		//CUDA_CALL(cudaMemcpy(mgx, data->cu_gradientX,MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		//CUDA_CALL(cudaMemcpy(mgy, data->cu_gradientY,MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		//for (int kk=0;kk<50;kk++)
		//{
		//	cout<<mgx[kk]<<" ";
		//}
		//cout<<endl;

		//for (int kk=0;kk<50;kk++)
		//{
		//	cout<<mgy[kk]<<" ";
		//}
		//cout<<endl;



		/*float *mwgx=new float[MAXIMUMPOINTDIM];
		float *mwgy=new float[MAXIMUMPOINTDIM];
		CUDA_CALL(cudaMemcpy(mwgx, data->cu_WarpedGradientX,MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));
		CUDA_CALL(cudaMemcpy(mwgy, data->cu_WarpedGradientY,MAXIMUMPOINTDIM*sizeof(float), cudaMemcpyDeviceToHost ));

		cout<<"gradient X\n";
		int cnum;
		for (int kk=0;kk<50;kk++)
		{
			cnum=inv_mask[kk][1]*data->t_width+inv_mask[kk][0];
			cout<<mwgx[cnum]<<" ";
		}
		cout<<endl;
		cout<<"gradient Y\n";
		for (int kk=0;kk<50;kk++)
		{
			cnum=inv_mask[kk][1]*data->t_width+inv_mask[kk][0];
			cout<<mwgy[cnum]<<" ";
		}
		cout<<endl;*/




		//shared memory
	
	/*	getJacobians_shared<<<grid,threads>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);*/
		/////////////////////////////////////////////////////////////////////////


		//get Jacobians grid without shared memory
		/*getJacobians_fullParral<<<grid,threads>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);*/

		//cout<<"Jacobians in\n";
		//worked version without constant memory fastest
		getJacobians<<<(data->t_width*data->t_height)/32+1,32>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);
		

		//Jacobians with texture memory, not correct and not as fast as the original one
	/*	getJacobians_TEX<<<(data->t_width*data->t_height)/32+1,32>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);*/
		
		
			
		//////////////////////Jacobians checked///////////////////////////
	/*	float *cJacobians=new float[allDim*data->t_width*data->t_height];

		CUDA_CALL(cudaMemcpy(cJacobians, data->cu_Jacobians, allDim*data->t_width*data->t_height*sizeof(float), cudaMemcpyDeviceToHost ));		
		ofstream out_J("Jacpbians_GPU.txt",ios::out);
		for (int kk=0;kk<2;kk++)
		{
			for (int jj=0;jj<data->s_dim+data->t_dim+4;jj++)
			{
				out_J<<cJacobians[kk*allDim+jj]<<" ";
			}
			out_J<<endl;
		}
		out_J.close();
		break*/;

	/*	float *cWarpTabel=new float[MAXIMUMPOINTDIM*3];
		CUDA_CALL(cudaMemcpy(cWarpTabel, data->cu_warp_tabel,MAXIMUMPOINTDIM*3*sizeof(float), cudaMemcpyDeviceToHost ));
		float *cJacobians=new float[allDim*data->t_width*data->t_height];
		
		CUDA_CALL(cudaMemcpy(cJacobians, data->cu_Jacobians, allDim*data->t_width*data->t_height*sizeof(float), cudaMemcpyDeviceToHost ));		
		cout<<"Jacobians\n";
		int ccnum=inv_mask[0][1]*data->t_width+inv_mask[0][0];
		for (int kk=0;kk<50;kk++)
		{
			cout<<cJacobians[0*allDim+kk]<<" ";
		}*/
	
	/*	bool isstop=false;
		for (int i=0;i<data->t_width;i++)
		{
			for (int j=0;j<data->t_height;j++)
			{
				if (cWarpTabel[j*data->t_width+i]!=-1)
				{

					for (int kk=0;kk<50;kk++)
					{
						cout<<cJacobians[(j*data->t_width+i)*allDim+kk]<<" ";
					}
					cout<<endl;
					isstop=true;
					break;
				}				
			}
			if (isstop)
			{
				break;
			}
		}
		cout<<endl;*/
		//float *m=new float[3*4];
		//float *n=new float[3];
		//for (int i=0;i<3;i++)
		//{
		//	n[i]=5+i;
		//	for (int j=0;j<4;j++)
		//	{
		//		m[i*4+j]=i*4+j+1;
		//	}
		//}
		//float *cu_m,*cu_n;
		//float *cu_result;
		//CUDA_CALL( cudaMalloc(&cu_m, 12 * sizeof(float)) );
		//CUDA_CALL(cudaMemcpy(cu_m,m,12*sizeof(float),cudaMemcpyHostToDevice));
		//CUDA_CALL( cudaMalloc(&cu_n, 4 * sizeof(float)) );
		//CUDA_CALL(cudaMemcpy(cu_n,n,3*sizeof(float),cudaMemcpyHostToDevice));
		//CUDA_CALL( cudaMalloc(&cu_result, 20 * sizeof(float)) );
		//alpha=1;beta=0;
		//CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,4,3,&alpha,cu_m,4,cu_n,incx,&beta,
		//	cu_result,incy));

		//float *result=new float[20];
		//CUDA_CALL(cudaMemcpy(result, cu_result,20*sizeof(float), cudaMemcpyDeviceToHost ));
		//for (int i=0;i<4;i++)
		//{
		//	cout<<result[i]<<" ";
		//}
		//cout<<endl;
		//break;

		
		

		//MatrixMVector(data->cu_Jacobians,data->pix_num,allDim,data->cu_errorImage,data->cu_deltaParameters,data);
	//	compactError2Full<<<(data->t_width*data->t_height+128)/128,128>>>(data->cu_errorImage,data->cu_fullErrorImage,data->cu_MaskTabel,data->t_width,data->t_height);
		alpha=1;beta=0;
		

		CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,allDim,data->pix_num,&alpha,data->cu_Jacobians,allDim,data->cu_errorImage,incx,&beta,
			data->cu_deltaParameters,incy));
		//break;
		///////////////////////J'E checked////////////////////////////////
		
		
		//float *cJEParameters=new float[MPD];
		//CUDA_CALL(cudaMemcpy(cJEParameters, data->cu_deltaParameters,MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"JE\n";
		//for (int kk=0;kk<allDim;kk++)
		//{
		//	cout<<cJEParameters[kk]<<" ";
		//}
		//cout<<endl;

		

		//getHessian
		CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
			allDim, allDim, data->pix_num, 
			&alpha, data->cu_Jacobians, allDim, 
			data->cu_Jacobians, allDim,
			&beta,
			data->cu_Hessian, allDim) );
		

		///////////////////////Hessian Checked////////////////////////////

	/*	CULA_CALL( culaDeviceSgetrf(allDim, allDim, data->cu_Hessian, allDim, ipiv) );
		CULA_CALL( culaDeviceSgetri(allDim, data->cu_Hessian, allDim, ipiv) );*/

		CUDA_CALL(cudaMemcpy(data->host_Hessian, data->cu_Hessian,allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
		invHessian(data->host_Hessian,data->host_inv_Hessian,allDim);
		CUDA_CALL(cudaMemcpy(data->cu_Hessian,data->host_inv_Hessian,allDim*allDim*sizeof(float),cudaMemcpyHostToDevice));


		////////////////////////////inv checked//////////////////////////////////////////
		//float *inv_hessian;
		//CUDA_CALL(cudaMalloc(&inv_hessian,allDim*allDim*sizeof(float)));
		//matrixInvert(data->cu_Hessian,inv_hessian,allDim);
		
		//try to solve inv in CPU
		//break;

	/*	cout<<"Hessian\n";
		for (int i=0;i<50;i++)
		{
			cout<<data->host_Hessian[i]<<" ";
		}
		cout<<endl;

		cout<<"inv_Hessian\n";
		for (int i=0;i<50;i++)
		{
			cout<<data->host_inv_Hessian[i]<<" ";
		}
		cout<<endl;*/
		
		


		//get the delta
		CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,allDim,allDim,&alpha,data->cu_Hessian,allDim,data->cu_deltaParameters,1,&beta,
			data->cu_deltaParameters,1));

		

		//float *cDeltaParameters=new float[MPD];
		//CUDA_CALL(cudaMemcpy(cDeltaParameters, data->cu_deltaParameters,MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"deltaP\n";
		//for (int kk=0;kk<allDim;kk++)
		//{
		//	cout<<cDeltaParameters[kk]<<" ";
		//}
		//cout<<endl;
		//

		//float *cParameters=new float[MPD];
		//CUDA_CALL(cudaMemcpy(cParameters, data->cu_parameters,MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		//cout<<"parameters before\n";
		//for (int kk=0;kk<allDim;kk++)
		//{
		//	cout<<cParameters[kk]<<" ";
		//}
		//cout<<endl;

		alpha=-1.0;
		//update parameters
		CUBLAS_CALL(cublasSaxpy_v2(data->blas_handle_,allDim,&alpha,data->cu_deltaParameters,1,data->cu_parameters,1));



	/*	CUDA_CALL(cudaMemcpy(cParameters, data->cu_parameters,MPD*sizeof(float), cudaMemcpyDeviceToHost ));
		cout<<"parameters after\n";
		for (int kk=0;kk<allDim;kk++)
		{
			cout<<cParameters[kk]<<" ";
		}
		cout<<endl;*/



		//check if we need to stop
		//CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->s_dim,data->cu_deltaParameters,incx,data->cu_deltaParameters,incy,&difference));
		//difference/=data->pix_num;
		//cout<<"iteration num: "<<times<<endl;
	//	cout<<"errorSum: "<<errorSum<<endl;
		//cout<<"error difference:"<<abs(lastError-errorSum)<<endl;
		//if (times>MaxIterNum||errorSum<0.2&&abs(lastError-errorSum)<0.001)
		if (times>MaxIterNum||errorSum<0.1&&abs(lastError-errorSum)<0.00001)
		{
			cout<<"time "<<times<<endl;
			float *parameters=new float[MPD];
			CUDA_CALL(cudaMemcpy(parameters, data->cu_parameters, MPD*sizeof(float), cudaMemcpyDeviceToHost ));
			checkIterationResult(parameters,data->ptsNum,data->s_dim,data->t_dim,true);
			delete []parameters;
			break;
		}

		
		//CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->s_dim,data->cu_currentShape,1,data->cu_lastShape,1));
		lastError=errorSum;
		//cout<<"errorSum: "<<errorSum<<endl;
		times++;

	

		//if (times==1)
		//{
		//		break;
		//}

	

		
	}

	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"time: "<<elapsedTime<<"/"<<times<<" ms"<<endl;


}

//int main(void)
//{
//	/*double **tmp;
//	tmp=new double*[3];
//	int i,j;
//	for(i=0;i<3;i++)
//	{
//		tmp[i]=new double[4];
//		for(j=0;j<4;j++)
//		{
//			tmp[i][j]=i*4+j;
//		}
//	}
//
//	float *cu_vec;
//	CUDA_CALL(cudaMalloc((void **)&cu_vec,MCN*sizeof(float)));
//	Mat2GPUVEC(tmp,3,4,cu_vec);
//
//	float *host_result_vec=new float[MCN*sizeof(float)];
//	CUDA_CALL(cudaMemcpy(host_result_vec,cu_vec,MCN*sizeof(float),cudaMemcpyDeviceToHost));
//	
//	for(i=0;i<3;i++)
//	{
//		for(j=0;j<4;j++)
//		{
//			cout<<host_result_vec[i*4+j];
//		}
//		cout<<endl;
//	}
//	cout<<1<<endl;*/
//	//testFunction();
//	return 0;
//}
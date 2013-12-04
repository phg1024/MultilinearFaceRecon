#include "imgAli_common.h"

////x-kernel
//[-1 0 1
//-2 0 2
//-1 0 1]
////y-kernel
//[-1 -2 -1
//0 0 0
//1 2 1
//]

ImgAli_common::ImgAli_common()
{
	x_kernel=cvCreateMat(3,3,CV_64FC1);
	y_kernel=cvCreateMat(3,3,CV_64FC1);
	//CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=-1;
	//CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=1;
	//CV_MAT_ELEM(*x_kernel,double,1,0)=-2;CV_MAT_ELEM(*x_kernel,double,1,2)=2;
	//CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;

	CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=0;
	CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=0;
	CV_MAT_ELEM(*x_kernel,double,1,0)=-0.5;CV_MAT_ELEM(*x_kernel,double,1,2)=0.5;
	CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;
	cvTranspose(x_kernel,y_kernel);

	//for(int i=0;i<3;i++)
	//{
	//	for(int j=0;j<3;j++)
	//	{
	//		CV_MAT_ELEM(*x_kernel,double,i,j)/=8;
	//		CV_MAT_ELEM(*y_kernel,double,i,j)/=8;
	//	}
	//}
}

void ImgAli_common::gradient(CvMat *img,CvMat *g_x,CvMat *g_y,CvMat *margnin)
{

	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		if(i>0&&i<width-1&&j>0&&j<height-1)
	//		{
	//			cvmSet(g_x,i,j)=((double)cvGet2D(img,i,j+1)-(double)cvGet2D(img,i,j-1))/2;
	//		}
	//	}
	//}

//	CvMat *imgSat=cvCreateMat(width,)
	cvFilter2D(img,g_x,x_kernel,cvPoint(-1,-1));  
	cvFilter2D(img,g_y,y_kernel,cvPoint(-1,-1));  

	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		if (CV_MAT_ELEM(*margnin,double,j,i)==1)
	//		{
	//			CV_MAT_ELEM(*g_x,double,j,i)=0;
	//			CV_MAT_ELEM(*g_y,double,j,i)=0;
	//		}
	//	}
	//}

	for(int i=0;i<cvGetSize(img).height;i++)
	{
		CV_MAT_ELEM(*g_x,double,i,0)*=2;
		CV_MAT_ELEM(*g_x,double,i,cvGetSize(img).width-1)*=2;
	}

	for(int i=0;i<cvGetSize(img).width;i++)
	{
		CV_MAT_ELEM(*g_y,double,0,i)*=2;
		CV_MAT_ELEM(*g_y,double,cvGetSize(img).height-1,i)*=2;
	}
}

void ImgAli_common::gradient(IplImage *img,CvMat *g_x,CvMat *g_y,CvMat *margnin)
{
	
	cvFilter2D(img,g_x,x_kernel,cvPoint(-1,-1));  
	cvFilter2D(img,g_y,y_kernel,cvPoint(-1,-1));  


	for(int i=0;i<cvGetSize(img).height;i++)
	{
		CV_MAT_ELEM(*g_x,double,i,0)*=2;
		CV_MAT_ELEM(*g_x,double,i,cvGetSize(img).width-1)*=2;
	}

	for(int i=0;i<cvGetSize(img).width;i++)
	{
		CV_MAT_ELEM(*g_y,double,0,i)*=2;
		CV_MAT_ELEM(*g_y,double,cvGetSize(img).height-1,i)*=2;
	}
}

void ImgAli_common::setTemplate(char *name)
{
	IplImage *tmp=cvLoadImage(name,0);
	Template=cvCreateMat(tmp->height,tmp->width,CV_64FC1);
	cvConvert(tmp,Template);

	//cvNamedWindow("1");
	//
	//cvDrawLine(Template,cvPoint(0,0),cvPoint(50,50),cvScalar(255,0,0));
	//cvShowImage("1",Template);
	//cvWaitKey();

	width=Template->cols;
	height=Template->rows;

	/*SD_ic=new double**[width];
	for (int i=0;i<width;i++)
	{
		SD_ic[i]=new double *[height];
		for (int j=0;j<height;j++)
		{
			SD_ic[i][j]=new double[shapeDim];
		}
	}*/

	//
	setParameters();
}

void ImgAli_common::setShapeDim(int dim)
{
	shapeDim=dim;
	//put the sd_ic in the paticular implementation
	//SD_ic=new double**[width];
	//for (int i=0;i<width;i++)
	//{
	//	SD_ic[i]=new double *[height];
	//	for (int j=0;j<height;j++)
	//	{
	//		SD_ic[i][j]=new double[shapeDim];
	//	}
	//}
}

void ImgAli_common::setTemplate(CvMat *img)
{
	width=img->cols;
	height=img->rows;
	Template=cvCreateMat(height,width,img->type);
	cvCopy(img,Template);
	//cvNamedWindow("1");
	//
	//cvDrawLine(Template,cvPoint(0,0),cvPoint(50,50),cvScalar(255,0,0));
	//cvShowImage("1",Template);
	//cvWaitKey();

	//width=Template->width;
	//height=Template->height;
	//
	setParameters();
}

void ImgAli_common::setParameters()
{	
	gradient_Ix=cvCreateMat(height,width,CV_64FC1);
	gradient_Iy=cvCreateMat(height,width,CV_64FC1);

	gradient_Tx=cvCreateMat(height,width,CV_64FC1);
	gradient_Ty=cvCreateMat(height,width,CV_64FC1);

	xyindex=cvCreateMat(3,width*height,CV_64FC1);
	//xmask=cvCreateMat(3,width*height,CV_64FC1);
	//ymask=cvCreateMat(3,width*height,CV_64FC1);
	//for (int i=0;i<height;i++)
	//{
	//	for (int j=0;j<width;j++)
	//	{
	//		CV_MAT_ELEM(*xyindex,double,0,i*height+j)=i;
	//		CV_MAT_ELEM(*xyindex,double,1,i*height+j)=j;
	//		CV_MAT_ELEM(*xyindex,double,2,i*height+j)=1;
	//		CV_MAT_ELEM(*xmask,double,0,i*height+j)=CV_MAT_ELEM(*xmask,double,1,i*height+j)=
	//			CV_MAT_ELEM(*xmask,double,2,i*height+j)=0;
	//		CV_MAT_ELEM(*ymask,double,0,i*height+j)=CV_MAT_ELEM(*ymask,double,1,i*height+j)=
	//			CV_MAT_ELEM(*ymask,double,2,i*height+j)=0;
	//	}
	//}
	//for (int i=0;i<width*height;i++)
	//{
	//	CV_MAT_ELEM(*xmask,double,0,i)=1;
	//	CV_MAT_ELEM(*ymask,double,1,i)=1;
	//}
	
	//WarpedInput=cvCreateImage(cvSize(Template->cols,Template->rows),Template->depth,Template->nChannels);
//	errorImage=cvCreateImage(cvGetSize(Template),Template->depth,Template->nChannels);
	//errorImageMat=cvCreateMat(Template->height,Template->width,CV_64FC1);
	//WarpedInput_mat=cvCreateMat(Template->height,Template->width,CV_64FC1);
	//Template_mat=cvCreateMat(Template->height,Template->width,CV_64FC1);
	//cvConvert(Template,Template_mat);
}

void ImgAli_common::setInput(char *name)
{
	Input = cvLoadImage(name,0);
}


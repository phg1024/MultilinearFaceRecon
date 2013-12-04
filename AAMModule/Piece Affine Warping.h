#ifndef PIECEAFFINEWARPING_H
#define PIECEAFFINEWARPING_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 

#include "shape.h"
//#include "delaunay triangulation.h"
#include <iostream>
#include <omp.h>
using namespace std;
using namespace cv;

class PieceAffineWarpping
{
//	void setTriangulation(Shape *,CvSubdiv2D *current,CvSubdiv2D *ref);

	//
	//IplImage *
	public:
		PieceAffineWarpping();
	IplImage * piecewiseAffineWarping(IplImage *,IplImage *,Shape *src,Shape *dst,CvMat *triangleList,bool drawn,affineParameters ***affineTable);
	CvMat * PieceAffineWarpping::piecewiseAffineWarping(CvMat *img,CvMat *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,affineParameters ***affineTable);

	int lastTriangleInd;

	bool getWeights(CvMat *point,Shape *dst,CvMat *triangleList,int k,double &alpha,double &beta,double &gamma);

	Mat m_dstImg,m_img;
	bool Interpolation;
	
	IplImage * piecewiseAffineWarping(IplImage *,IplImage *,Shape *src,Shape *dst,CvMat *triangleList,bool drawn,double **,int **);
	CvMat * PieceAffineWarpping::piecewiseAffineWarping(CvMat *img,CvMat *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,double **,int **);


	IplImage * piecewiseAffineWarping_GPU(IplImage *,IplImage *,Shape *src,Shape *dst,CvMat *triangleList,bool drawn,double **,int **);
};

#endif
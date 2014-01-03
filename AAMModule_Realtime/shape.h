#ifndef SHAPE_H
#define SHAPE_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 

#include <string>
#include <fstream>
#include "saveandread.h"
using namespace std;
using namespace cv;

struct affineParameters{
	int triangleInd;
	double alpha,beta,gamma;
};

class Shape
{
	public:
		Shape();
		Shape(Shape *);
		
		~Shape();
		int ptsNum;
		double **pts;
		double *ptsForMatlab;//also the normalized shape
		int triangleNum;
		double **triangles;

		double width,height;
		double minx,miny,maxx,maxy;

		void setPtsNum(int num);
		void setTriangleNum(int num);

		void setPts(int,double **);
		void setTriangles(int, double **);

		
		IplImage * hostImage;
		void setHostImage(IplImage *img);

		void getVertex(string imgName);
		void getVertex(double *,int ptsNum,int width,int height);
		void centerPts(int);
		void normalize(int);
		void scale(double,int,bool centered=false);
		double scaleParameter;

		double *getcenter();
		
		void translate(double tx,double ty,int stype=1);
		void getMask(CvMat *triangleList);
		int pix_num;//only for meanshape, the number of pixels in mesh
		CvMat *mask;
		CvMat *mask_withindex;

		void getPtsIndex();
		double **pts_Index;
		int **inv_mask;

		CvMat *marginMask;
		affineParameters ***affineTable;
		affineParameters ****affineTable_strong;
		void getTabel(CvMat *triangleList);
		void getTabel_strong(CvMat *triangleList);
		void getMargin();

		//save the shape
		//SL_Basis *sl_engine;
		void save(string name);
		void save(ofstream &);
		void load(istream &in);

		void init();

		Mat colorImg;
		//CvSparseMat colorImg;

		void getDpeth(string ptsName,int trainStyle=0);

		double ** weightTabel;
		//	Mat indexTabel;
		void getWeightTabel(affineParameters ***affineTable);

};




#endif
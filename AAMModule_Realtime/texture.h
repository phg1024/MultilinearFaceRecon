#ifndef TEXTURE_H
#define TEXTURE_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"
#include "engine.h"
//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 
#include <iostream>
using namespace std;
using namespace cv;



class Texture
{
public:
	Texture(IplImage *,CvMat *);
	Texture();
	void setImg(IplImage *);
	CvMat *mask;
	void getROI(IplImage *,CvMat *);
	CvMat *imgData;
	int width,height;
	int row,col;
	int nband;
	IplImage *img;
	IplImage *NormailizedImg;
	double normalize();
	double simple_normalize();
	int pix_num;

	IplImage* getImage();

	void setZero();

	CvMat *getImageMat();

	int depth,nchanels;//depth and nchannels for texture

	double pointMul(Texture *);
	void devide(double);
	void texture_add(Texture *);
	void texture_deduce(Texture *);
	friend Texture &operator +(Texture &,Texture &);
	friend Texture &operator +(Texture &,double w);
	friend Texture &operator -(Texture &,Texture &);
	Texture &operator =(Texture &);
	friend double operator *(Texture &,Texture &);
	friend Texture &operator /(Texture &,double );
	friend Texture& operator *(Texture &in1,double w);


	//save
	void save(ofstream &);
	void load(istream &);


	//for cv
	Mat m_imgData;
	Mat m_mask;

	
	
};




#endif
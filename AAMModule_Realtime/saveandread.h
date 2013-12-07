#ifndef SAVEANDREAD_H
#define SAVEANDREAD_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"
//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 
//#pragma comment(lib,"mclcommain.lib") 
//#pragma comment(lib,"mclxlmain.lib") 
//#pragma comment(lib,"libeng.lib") 
//#pragma comment(lib,"libmx.lib")
//#pragma comment(lib,"libmat.lib")

#include <iostream>
#include <fstream>
using namespace std;

class SL_Basis
{
public:
	// if I delete the data, it is possible to lose the data.
	// since I read it only for the first time, so will have no problem
	//but remember this, potential errors!
	void saveMatrix(ofstream &out,CvMat *mat);
	CvMat* loadMatrix(ifstream &in,CvMat *mat);

	void saveMatrix(ofstream &out,IplImage *img);
	IplImage* loadMatrix(ifstream &in,IplImage *img);

	void saveMatrix(ofstream& out,double *data,int length);
	double* loadMatrix(ifstream &in,double *data);

	void saveMatrix(ofstream& out,int *data,int length);
	int * loadMatrix(ifstream &in,int *data);

	void saveMatrix(ofstream& out,int **data,int s1,int s2);
	int ** loadMatrix(ifstream &in,int **data);

	void saveMatrix(ofstream& out,double **data,int s1,int s2);
	double ** loadMatrix(ifstream &in,double **data);
};

#endif
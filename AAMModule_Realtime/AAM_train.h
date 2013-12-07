#ifndef AAM_TRAIN_H
#define AAM_TRAIN_H
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"
#include "engine.h"
//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 
//#pragma comment(lib,"mclcommain.lib") 
//#pragma comment(lib,"mclxlmain.lib") 
#pragma comment(lib,"libeng.lib") 
#pragma comment(lib,"libmx.lib")
//#pragma comment(lib,"libmat.lib")

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#include "shape.h"
#include "AAM_common.h"
#include "texture.h"
#include "saveandread.h"

class AAM_Train
{
public:
	AAM_Train(string);
	void getTrainingdata(string dirName);
	Shape **shape;
	int shapeNum;
	Engine *ep;
	void getMeanShape();
	void getTexture();
	void getMeanTexture();
	Shape *meanShape;
	Texture *meantexure,*meantexure_real;
	Texture **texture;
	double normalize(double *,int ptsnum);

	void shape_pca();
	void texture_pca();
	int shape_dim,texture_dim;
	CvMat *s_vec,*t_vec,*s_mean,*t_mean;


	CvMat *s_value,*t_value;
		double texture_scale,shape_scale;
		AAM_Common *refShape;
	bool isGlobaltransform;
	void setGlobal(bool);

	//save the trained results
	SL_Basis *SL_engine;
	void saveResult();
	
	string dirName;
};

#endif
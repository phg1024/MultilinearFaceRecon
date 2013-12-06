#ifndef AAM_COMMON_H
#define AAM_COMMON_H

#define NOMINMAX
#include <Windows.h>


#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include "shape.h"
#include "delaunay triangulation.h"
#include "Piece Affine Warping.h"

class AAM_Common : public PieceAffineWarpping
{
	public:
		AAM_Common();
		~AAM_Common();
		Delaunay_Tri *ref;

		void setRef(double **);
		
		//PieceAffineWarpping *warp;

		//void train(char *ImahesetPath);
		
		//int ptsNum;
	protected:
	private:
};




#endif
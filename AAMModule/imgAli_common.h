#ifndef IMGALI_COMMON_H
#define IMGALI_COMMON_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"
#include "shape.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 
//define the common methods that will be used
class ImgAli_common
{
public:
	ImgAli_common();
	CvMat *Template;//template
	IplImage *Input,*WarpedInput;
	CvMat *errorImageMat,*WarpedInput_mat,*Template_mat;
	CvMat *x_kernel,*y_kernel;
	void gradient(CvMat *img,CvMat *g_x,CvMat *g_y,CvMat *margnin);
	void gradient(IplImage *img,CvMat *g_x,CvMat *g_y,CvMat *margnin);
	CvMat * gradient_Tx,*gradient_Ty;
	CvMat * gradient_Ix,*gradient_Iy;
	CvMat * Jacobian;
	CvMat * Hessian;
	CvMat *inv_Hessian;
	CvMat *steepImage;
	double ***SD_ic;
	int shapeDim;

	//initialize the matrices after we have the size of the template
	void setParameters();

	void setTemplate(char *);
	void setTemplate(CvMat *);
	void setInput(char *);
	void setShapeDim(int dim);
	int width,height;

	CvMat *xyindex;//original xy grid
	CvMat *xmask,*ymask;

protected:
private:
};


#endif
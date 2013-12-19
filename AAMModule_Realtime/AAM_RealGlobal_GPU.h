#ifndef AAM_REALGLOBAL_GPU_H
#define AAM_REALGLOBAL_GPU_H

//global transform using theta,k and t
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
//#include "opencv\cxerror.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
#pragma comment(lib,"winmm.lib") 
//#pragma comment(lib,"opencv_gpu230.lib") 


#include <string>
#include <fstream>
using namespace std;

#include "shape.h"
#include "AAM_common.h"
#include "texture.h"
#include "imgAli_common.h"
#include "AAM_train.h"
#include <omp.h>
#include <mmsystem.h>
#include "opencv2/gpu/gpu.hpp"
//#include "MM.h"
#include "randtree.h"
#include "aptPCA_float.h"


using namespace cv;
using namespace cv::gpu;


class AAM_RealGlobal_GPU: public ImgAli_common
{
public:
	AAM_RealGlobal_GPU(double _search_test=0,double AAM_weight=1,double _priorWeight=0,double _localPCAWeight=0,bool isApt=false);
	~AAM_RealGlobal_GPU();

	bool getCurrentStatus();
	void setCurrentStatus(bool);

	void setSaveName(char * name);
	void setGlobalStartNum(int);

	void getAllNeededData(AAM_Train *trainedResult);
	void search(IplImage *);
	void getCenter_Scale(IplImage *);

	Point offset; double s_scale;

	int shape_dim,texture_dim;
	CvMat *s_vec,*t_vec,*s_mean,*t_mean;
	CvMat *s_value,*t_value;
	double texture_scale,shape_scale;
	CvMat *triangleList;
	void getJacobian(Shape *,CvMat *triangleList);
	CvHaarClassifierCascade *face_cascade;
	CvMemStorage*  faces_storage;
	CascadeClassifier leye_cascade,reye_cascade,
		nose_cascade,mouth_cascade;
	Shape **shapes;
	Texture **textures;
	CvMat *mask;//texture
	int shapeWidth,shapeHeight;
	Shape *meanShape;
	Texture *meanTexture;
	int **triangleIndList;
	int *listNum;
	int pix_num;
	int nband;

	double **pts_Index;
	int **inv_mask;
	CvMat *mask_withindex;

	Point faceCenter;
	double face_scale;

	PieceAffineWarpping *warp;

	void searchVideo(string videoName);
	void searchPics(string picListName);
	void detect(IplImage *,double scale);
	void iterate(IplImage *Input);
	
	void precompute();

	////////////////////added variables////////////////////////
	affineParameters ***affineTable;
	affineParameters ****affineTable_strong;
	CvMat **g_ax,**g_ay;
	double ****full_Jacobian;
	double *s_weight,*t_weight;//current shape and texture weight
	Texture *curr_texture;
	void getCurrentTexture(double *t_weight);//get current texture according to the texture weights
	void getSD_sim();//get the SD_sim image
	void getHessian();
	CvMat *tmpSD;
	CvMat *curhessian,*newHessian;
	Texture *currentTemplate,*currentTexture;
	//Texture **Tex_vecs;

	void prepareForTracking();

	Texture *meantexture_real;

	//save current shape
	Shape *currentShape;
	int frameNum;

	//condider global transform?
	bool isGlobaltransform;

	void loadResult(string name);
	void getAllNeededData(string name);

	//the global transform
	double tran_x,tran_y;
	//double *last_Iteration_weight;

	//current base shapes
	CvMat *cur_XYShape;
	CvMat *cur_YXShape;
	CvMat *Cur_Global,*Cur_Global_inv;

	CvMat *s_vec_tran;
	double *s_weight_vec;


	////////////using namespace cv
	Mat m_hessian,m_inv_hessian;
	Mat m_mask,m_mask_withindex;
	Mat m_gradient_Tx,m_gradient_Ty;
	Mat *m_g_ax,*m_g_ay;
	Mat m_errorImageMat;
	Mat m_triangleList;
	Mat m_s_vec,m_t_vec;

	Mat HessianLocal;

	bool outputtime;

	int startNum;
	int currentFrame;

	float probForEachFeature[20];
	float probForEachFeatureCandidates[30][30];
	int candidateNum[30];
	float distanceKNN[20];
	/////////////////for hessian
	Mat  fullSD, fullSD_tran,full_Hessian;
	//GpuMat  fullSD_gpu;
	//GpuMat  fullSD_tran_GPU;
	//GpuMat gpu_Hessian;
	//	int dim;

	//for transform
	double theta,k_scale,transform_x,transform_y;
	CvMat *inputGradient_x,*inputGradient_y;
	CvMat *warp_igx,*warp_igy;
	Mat m_inputGradient_x,m_input_Gradient_y;
	Mat m_warp_igx,m_warp_igy;

	CvMat *mat_currentImage;
	Mat m_currentImage;
	double tex_scale;
	Shape *currentLocalShape;

	void setSmoothnessWeight(double);
	string dataDir;
	double smoothWeight;
	double smoothWeight_backup;
	double **ptsLast;
	double **SD_smooth;

	Mat  fullSD_smooth, fullSD_tran_smooth,full_Hessian_smooth;

	double Pts_Difference;

	double stepLength;

	Mat lastTemplate;

	double AAM_weight;//we could try to rely on optical flow
	int resizeSize;
	void setResizeSize(int);
	double lastErrorSum;
	bool showSingleStep;


	//step length adjustment
	double *s_weight_last,*t_weight_last;
	double k_scale_last,transform_x_last,transform_y_last,theta_last;
	bool recaculate;
	int increaseTime;
	double initialTx,initialTy;
	void setInitialTranslation(double _tx,double _ty)
	{
		initialTx=_tx;initialTy=_ty;
	}
	double initialScale;
	void setInitialScale(double _scale)
	{
		initialScale=_scale;
	}

	double initialTheta;
	void setInitialTheta(double _theta)
	{
		initialTheta=_theta;
	}

	int pNum; //the number of threads to caculate M*M'
	Mat *p_fullSD,*p_fullSD_tran,*p_fullHessian;
	int parallelStep;

	void setCUDA(bool _using)
	{
		usingCUDA=_using;
	}
	bool usingCUDA;
	int cuda_cols;
	int cuda_rows;
	int cuda_row_pitch;
	int cuda_numF;
	vector<float> cuda_data_cpu;

	//////PAW optimization///////////////
	int ** indexTabel;


	/////////////////for combination//////////////////
	void preProcess_GPU_combination();

	Mat *conv_precalculated;

	///////////////////////GPU//////////////////////////////////////
	void preProcess_GPU();
	void iterate_GPU(IplImage *Input);
	void setHostData(float*data ,Mat &M,int);
	float *parameters,*inputImg;
	float *cu_gradientX,*cu_gradientY;

	void iterate_clean(Mat &Input);
	bool iterate_cpu(Mat &input,int type);
	bool iterate_clean_CPU(Mat &,RandTree *_trees, int type=0);

	void calculateData_onrun_AAM_combination(Mat &m_img,vector<int> &ind,bool keepOriginalValue=false);
	//RandTree *trees;

	bool usingGPU;

	void setGPU(bool _using)
	{
		usingGPU=_using;
	}

	RandTree *trees;
	double ***SD_detection;
	bool isSDDefined;
	Mat  fullSD_detection, fullSD_tran_detection,full_Hessian_detection;
	bool useDetectionResults;

	//hardcoded to 15,15,32,13 currently
	//int 
	//string trainedTree;
	//void setTrainedTree(string _fileName)
	//{
	//	rt.usingCUDA=usingCUDA;
	//	rt.withAAM=true;
	//	rt.load(_fileName);
	//}
	RandTree rt;//(15,3,0,15,32,13);
	bool usingTrees;
	bool treeInitialization;
	double *meanShapeCenter;

	//int labelNum;//the number of labels


	void setUsingTrees(bool s)
	{
		usingTrees=s;
	}
//	int interestPtsInd[6];

	//robust error
	Mat imageMask;

	///////////////prior term and use the probability map/////////////////////
	double priorWeight;
	double priorWeight_backup;

	double localPCAWeight;
	double localPCAWeight_backup;

	Mat priorSigma;
	double *priorMean;//need to be read in
	double *SD_prior;
	double *SD_local;

	//calculate the probability covariance matrix for all the positions
	void calculateMandC(Mat &prob_map,double x, double y, int gridSize,double *mu,Mat &conv);
	void calculateMandC_withGuidance(Mat &prob_map,double dx, double dy, int gridSize,double *mu,Mat &conv,int *shapeG,int featureID,int candidateID=0);

	//pre-calculate, no need to do it when iteration
	void calculateMandC_preCalculate(Mat &prob_map, int gridSize,int id,int largeWindowSize);

	void calculateMandC_autoSized(Mat &prob_map,double x, double y, int gridSize,double *mu,Mat &conv);
	int **shapeSample;

	vector<Point> candidatePoints[50];
	//bool usePts[50][10];

	Mat *prob_conv;
	double **prob_mu;

	double ***prob_mu_candidates;
	Mat **prob_conv_candidates;

	void calculateTermValue(double errorSum,double &);

	string currentNamewithoutHouzhui;

	char prefix[500];

	//local PCA
	CvMat *local_s_vec,*local_s_mean;
	Mat m_local_s_mean;
	int local_shape_dim;
	Mat m_local_mean;
	Mat m_localHessian;

	Adp_PCA_float *textureModel;
	bool isAdaptive;
	MatrixXf curData;
	Mat textureEigenValues;
	int subjNum;
};





#endif
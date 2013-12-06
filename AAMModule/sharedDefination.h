#ifndef SHAREDDEFINATION_H
#define SHAREDDEFINATION_H

const int MPN=500000;
const int MAX_LABEL_NUMBER=30;

#define  MAX_COUNT_NUM 100000;
#define MAX_POINT_DIM 500;
#define MAX_PIXEL_NUM 500000;
#define BlockNumGlobal 10

extern "C" void invHessian(float *inputHessian, float *outputHessian,int dim);
extern "C" void checkIterationResult(float *parameters,int ptsNum,int s_dim,int t_dim,bool);

extern "C" void saveIterationResult(float *pts,int ptsNum);

extern "C" void solveAb(float *inputHessian, float *b,float *deltaX,int dim);
extern "C" void updateModelCPU(float *,int,float *);

struct Node
{
	Node()
	{
		l_child=r_child=NULL;
		sampleInd=new vector<int>;
		label=-1;
		num_of_each_class=NULL;
		num_all=0;
		pos1[0]=pos1[1]=pos2[0]=pos2[1]=0;
		threshold=0;
	}
	int pos1[2];
	int pos2[2];
	//float *posteriorProb;//only used for leaves
	int label;
	int nLevel;
	Node *l_child,*r_child;
	vector <int> *sampleInd;

	//only for leaves
	int num_all;
	float *num_of_each_class;
	float threshold;
};



struct LabelResult
{
	LabelResult()
	{
		label=-1;
		prob=0;
		//prob_all=new float[10];
	}

	int label;
	float prob;
	float prob_all[MAX_LABEL_NUMBER];
};

struct LeafNode
{
	LeafNode(){leaf_node=NULL;label=-1;};
	Node *leaf_node;
	int label;
};

//extern "C" void setData_preprocess(int _max_depth,int _min_sample_count,double _regression_accuracy, int _max_num_of_trees_in_the_forest,int _windowSize, int labelNum, Node **trees_cpu,int treeNum,bool withDepth=false);
extern "C" void predict_GPU(float *host_img,int width,int height,float *host_result);

extern "C" void predict_GPU_withDepth(float *colorImg, float *depthImg,int width,int height,float *host_result,int trainStyle);

extern "C" void predict_GPU_withDepth_clean(int width,int height,float *host_result,int trainStyle);

extern "C" void train_GPU();

extern "C" void aa(int rows,int cols,int row_pitch,int numF,vector<float> &data_cpu);
extern "C" void setData_Preprocess(float *s_vec,float *t_vec,float *s_mean,float *t_mean,
								   float *warpTabel,float *triangle_indexTabel,int s_dim,int t_dim,int ptsNum,int pix_num,
								   int t_width,int t_height,float *shapeJacobians,float *MaskTable,float *fowardIndex,bool showSingleStep);
extern "C" void setData_onRun(float *parameters,float *inputImage,float *InputGradientX,float *InputGradientY,int width,int height);
extern "C" void iterate_CUDA(int width,int height,double smoothWeight,double,int currentFrame,int startFrame,float *h,int **inv_mask);
extern "C" void init();
extern "C"  void endSection();
//extern "C" void setData_RT_onrun(float *colorImg,float *depthImg,int width,int height);

extern "C" void setData_onrun_shared(float *colorImg,float *depthImg,int width,int height);

//extern "C" void setCPUDetection2GPU(float *);

//copy calculated local prior mean and covariance matrix
extern "C" void setLocalPrior(float *mean,float *conv,int shapeDim,int textureDim);

#define MAX_COUNT 5000000


//for combination
extern "C" void predict_GPU_withDepth_combination(int width,int height,float *host_result,int trainStyle,int **finalPos,float *);

extern "C" bool predict_GPU_separated_combination(int width,int height,float *host_result,float **finalPos,float *,int,int,int,int,bool isInitial=false,bool showProb=false,float lastTheta=0);

//type=0: color model, type=1: depth model
extern "C" void setData_RandomizedTrees_combination(int _max_depth,int _min_sample_count,double _regression_accuracy, int _max_num_of_trees_in_the_forest,int _windowSize, int labelNum, Node **trees_cpu,int treeNum,bool withDepth=false,int type=0);

extern "C" void setData_onRun_AAM(float *parameters,int width,int height,vector<int>&finalInd);
extern "C" void setData_AAM_combination(float *s_vec,float *t_vec,float *s_mean,float *t_mean,
	float *warpTabel,float *triangle_indexTabel,int s_dim,int t_dim,int ptsNum,int pix_num,
	int t_width,int t_height,float *shapeJacobians,float *MaskTable,float *fowardIndex,bool showSingleStep,
	bool isAptive=false,float *dataAddress=NULL);

extern "C" void setShowSingle(bool);


//for AAM detection term
extern "C" void setupConv(int width, int height,int windowSize,int **shapeLoc,vector<int> &indList,float *);
extern "C" void setupConv_featurePts(int width, int height,int windowSize,float **shapeLoc,vector<int> &indList,float *absVisibleIndex);

extern "C" int iterate_combination(int width,int height,int currentFrame,int startFrame,float &theta,float *shape, int& shapePtsNum, bool isAAMOnly=false,bool showNN=false,bool updateOnly=false);
extern "C" void setupWeight(double AAMWeight,double RTweight,double priorWeight,double localWeight=0);

extern "C" void setupUsedIndex(int *,int);
extern "C" void setColorDepthData(float *colorImg,float *depthImg);
extern "C" void setAbsIndex(float *absInd);
extern "C" void setPMeanHessian(float *p_mean,float *p_sigma);
//extern "C" void checkProbMap(float *);
//extern "C" void solveAxB(float *A,float *b,int dim);
#endif
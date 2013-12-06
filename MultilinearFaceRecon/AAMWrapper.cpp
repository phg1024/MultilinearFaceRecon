#include "AAMWrapper.h"


AAMWrapper::AAMWrapper(void)
{
	setup();
}


AAMWrapper::~AAMWrapper(void)
{
}

void AAMWrapper::setup() {
	string datapath = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\model\\";
	string searchPicDir;
	string savePrefix;
	string AAMSearchPrefix;
	string colorRT_model;
	string depthRT_model;
	string AAMModelPath;
	string alignedShapeDir;

	searchPicDir=datapath + "imgList.txt";
	savePrefix="DougTalking_AAM_Sin";

	colorRT_model= datapath + "trainedTree_15_12_56_22_1.txt";
	depthRT_model= datapath + "trainedTree_15_12_64_22_0.txt";

	AAMModelPath= datapath + "trainedResult_90_90.txt";
	alignedShapeDir= datapath + "allignedshape_90_90.txt";

	engine=new AAM_Detection_Combination(1,0.005,0.001,0,colorRT_model,depthRT_model,AAMModelPath,alignedShapeDir,true);
}

vector<float> AAMWrapper::track( const unsigned char* cimg, const unsigned char* dimg, int w, int h )
{
	return engine->trackWithData(cimg, dimg, w, h);
}

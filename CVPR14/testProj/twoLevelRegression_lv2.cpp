#include "TwoLevelRegression_lv2.h"

void TwoLevelRegression_LV2::loadFull(char *modelLV1,char *modelLV2)
{
	loadFerns_bin(modelLV1);
	model_lv2.loadFerns_bin(modelLV2);
}

bool TwoLevelRegression_LV2::predict_real_lv2(IplImage *img,int sampleNum)
{
	d.showFace=showRes;
	vector<Rect> faceRegionList=d.findFaceFull(img);

	vector<Mat> facesMat= predict_real_givenRects_L2(img,faceRegionList, model_lv2,sampleNum);

	if(facesMat.size()==0)
		return false;
	else
		return true;
}
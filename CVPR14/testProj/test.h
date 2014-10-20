#pragma once
#include "TwoLevelRegression.h"
#include "codetimer.h"
#include "Evaluation.h"
#include "FaceFounder.h"

class Test
{
public:
	void getScaleTranslation(char *modelName,char *fileListName);

	int isAFace(vector<Rect> &,Shape &);
	bool isCollide(Rect &a,Rect &b);
};
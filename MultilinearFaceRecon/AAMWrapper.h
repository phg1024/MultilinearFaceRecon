#pragma once

#include "phgutils.h"
#include "../AAMModule_Realtime/AAM_Detection_Combination.h"
#include "Utils/Timer.h"

// an interface layer for AAM

class AAMWrapper
{
public:
	AAMWrapper(void);
	~AAMWrapper(void);

	void reset();
	// tracking interface
	const vector<float>& track(const unsigned char* cimg, const unsigned char* dimg, int w, int h);

	float getPrepTime() const {
		return tPrep.elapsed();
	}

protected:
	void setup();

private:
	// input data
	Mat colorImage, colorIMG_Gray, depthImg;

	// flag: whether initial fitting or not
	bool initial;
	bool isRecording;
	// bounding box of the face region
	int startX, endX, startY, endY;

	AAM_Detection_Combination *engine;

	// engine related parameters
	int startNum;	// not sure what it is
	int curStatus;

	float lastShape[200], currentShape[200];
	vector<float> ptsList;

	// returned points
	vector<float> f;
	vector<float> eptf;	// dummy

	PhGUtils::Timer tPrep;
};


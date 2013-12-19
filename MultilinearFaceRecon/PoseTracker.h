#pragma once

#include "MultilinearReconstructor.h"
#include "AAMWrapper.h"
#include "Utils/Timer.h"

class PoseTracker
{
public:
	PoseTracker(void);
	~PoseTracker(void);

	bool reconstructionWithSingleFrame(
		const unsigned char* colordata,
		const unsigned char* depthdata,
		vector<float>& pose,
		vector<float>& fpts
	);

private:
	bool loadLandmarks();
	void bindTargetLandmarks( const vector<PhGUtils::Point3f>& lms );
	void fit();

private:
	vector<int> landmarks;
	MultilinearReconstructor recon;

	int frameIdx;
	vector<PhGUtils::Point3f> lms;		// landmarks got from AAM tracking
	AAMWrapper aam;
};


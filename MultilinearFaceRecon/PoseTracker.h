#pragma once

#include "MultilinearReconstructor.h"
#include "AAMWrapper.h"
#include "Utils/Timer.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"

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

	void printStats() {
		PhGUtils::message("Average tracking prep time = " + PhGUtils::toString(aam.getPrepTime() / trackedFrames * 1000) + "ms");
		PhGUtils::message("Average tracking time = " + PhGUtils::toString(tAAM.elapsed() / trackedFrames * 1000) + "ms");
		PhGUtils::message("Average reconstruction time = " + PhGUtils::toString(tRecon.elapsed() / frameIdx * 1000) + "ms");
		PhGUtils::message("Average other time = " + PhGUtils::toString(tOther.elapsed() / frameIdx * 1000) + "ms");
	}

private:
	bool loadLandmarks();
	void bindTargetLandmarks( const vector<PhGUtils::Point3f>& lms );
	void fit();

private:
	vector<int> landmarks;
	vector<pair<PhGUtils::Point3f, int>> labeledLandmarks;

	MultilinearReconstructor recon;

	int trackedFrames, frameIdx;
	vector<PhGUtils::Point3f> lms;		// landmarks got from AAM tracking
	AAMWrapper aam;

	PhGUtils::Timer tAAM, tRecon, tBinding, tOther;
};


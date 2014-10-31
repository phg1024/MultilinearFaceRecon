#pragma once

#include "MultilinearReconstructor_old.h"
#include "MultilinearReconstructorGPU.cuh"

#include "AAMWrapper.h"
#include "Utils/Timer.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"

class PoseTracker
{
public:
	PoseTracker(void);
	~PoseTracker(void);

	void reset();

	float facialFeatureTrackingError() const;
	float poseEstimationError() const;

	bool reconstructionWithSingleFrame(
		const unsigned char* colordata,
		const unsigned char* depthdata,
		vector<float>& pose,
		vector<float>& fpts
	);

	void printStats() const {
		//PhGUtils::message("Average tracking prep time = " + PhGUtils::toString(aam.getPrepTime() / trackedFrames * 1000) + "ms");
		PhGUtils::message("Setup time = " + PhGUtils::toString(tSetup.elapsed() * 1000) + "ms");
		PhGUtils::message("Average tracking time = " + PhGUtils::toString(tAAM.elapsed() / trackedFrames * 1000) + "ms");
		PhGUtils::message("Average reconstruction time = " + PhGUtils::toString(tRecon.elapsed() / frameIdx * 1000) + "ms");
		PhGUtils::message("Average other time = " + PhGUtils::toString(tOther.elapsed() / frameIdx * 1000) + "ms");
		PhGUtils::message("Average total time = " + PhGUtils::toString(tTotal.elapsed() / frameIdx * 1000) + "ms");
	}

	const Tensor1<float>& getMesh() const {
		return reconGPU.currentMesh();
	}

	const PhGUtils::QuadMesh& getQuadMesh() const {
		return mesh;
	}

private:
	bool loadLandmarks();
	void fit();

private:
	vector<int> landmarks;
	vector<pair<PhGUtils::Point3f, int>> labeledLandmarks;

	MultilinearReconstructorGPU reconGPU;
	MultilinearReconstructor_old recon;	

	int trackedFrames, frameIdx;
	vector<PhGUtils::Point3f> lms;		// landmarks got from AAM tracking
	AAMWrapper* aam;

	PhGUtils::Timer tAAM, tRecon, tSetup, tOther, tTotal;
	PhGUtils::QuadMesh mesh;
};


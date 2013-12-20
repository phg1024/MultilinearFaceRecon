#include "PoseTracker.h"

#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/Timer.h"

#include "Kinect/KinectUtils.h"

PoseTracker::PoseTracker(void)
{
	loadLandmarks();

	lms.resize(128);
	labeledLandmarks.resize(128);

	frameIdx = 0;
	trackedFrames = 0;
}


PoseTracker::~PoseTracker(void)
{
}

bool PoseTracker::loadLandmarks()
{
	const string filename = "../Data/landmarks.txt";
	ifstream fin(filename, ios::in);
	if( fin.is_open() ) {
		landmarks.reserve(128);
		int idx;
		while(fin.good()) {
			fin >> idx;
			landmarks.push_back(idx);
		}
		PhGUtils::message("landmarks loaded.");
		cout << "total landmarks = " << landmarks.size() << endl;

		return true;
	}
	else {
		return false;
	}
}

void PoseTracker::bindTargetLandmarks( const vector<PhGUtils::Point3f>& lms )
{
	for(int i=0;i<landmarks.size();i++) {
		int vidx = landmarks[i];
		labeledLandmarks[i] = (make_pair(lms[i], vidx));
	}
	recon.bindTarget(labeledLandmarks);
}

bool PoseTracker::reconstructionWithSingleFrame(
	const unsigned char* colordata,
	const unsigned char* depthdata,
	vector<float>& pose,
	vector<float>& fpts
	) 
{
	const int w = 640, h = 480;

	// AAM tracking
	tAAM.tic();
	trackedFrames++;
	fpts = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	if( fpts.empty() ) {
		// tracking failed
		cerr << "AAM tracking failed." << endl;
		return false;
	}
	else {
		tOther.tic();
		// get the 3D landmarks and feed to recon manager
		int npts = fpts.size()/2;
		for(int i=0, j=npts;i<npts;i++,j++) {
			int u = fpts[i];
			// flip y coordinates
			int v = fpts[j];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			PhGUtils::colorToWorld(u, v, d, lms[i].x, lms[i].y, lms[i].z);
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);

			// minus one is a hack to bring the model nearer
			//lms[i].z += 1.0;
		}
		bindTargetLandmarks(lms);
		tOther.toc();

		tRecon.tic();
		if( frameIdx++ == 0 ) {
			// fit the pose first, then fit the identity and pose together
			recon.fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		}
		else{
			recon.fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
		}
		tRecon.toc();

		tOther.tic();
		pose.assign(recon.getPose(), recon.getPose()+7);
		tOther.toc();

		return true;
	}
}
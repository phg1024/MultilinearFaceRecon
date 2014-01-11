#include "PoseTracker.h"

#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/Timer.h"

#include "Kinect/KinectUtils.h"

PoseTracker::PoseTracker(void)
{
	// set the template mesh
	PhGUtils::OBJLoader loader;
	loader.load("../Data/shape_0.obj");
	mesh.initWithLoader( loader );
	recon.setBaseMesh(mesh);

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
	const string filename = "../Data/model/landmarks.txt";
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
	recon.bindTarget(labeledLandmarks, MultilinearReconstructor::TargetType_2D);
}

void PoseTracker::bindRGBDImage(const vector<unsigned char>& colordata, const vector<unsigned char>& depthdata)
{
	recon.bindRGBDTarget(colordata, depthdata);
}


bool PoseTracker::reconstructionWithSingleFrame(
	const unsigned char* colordata,
	const unsigned char* depthdata,
	vector<float>& pose,
	vector<float>& fpts
	) 
{
	const int w = 640, h = 480;

	tTotal.tic();

	// AAM tracking
	tAAM.tic();
	trackedFrames++;
	fpts = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	if( fpts.empty() ) {
		// tracking failed
		cerr << "AAM tracking failed." << endl;
		tTotal.toc();
		return false;
	}
	else {
		tOther.tic();
		// get the 3D landmarks and feed to recon manager
		int mfilterSize = 3;
		int neighbors[] = {-1, 0, 1};
		int npts = fpts.size()/2;
		for(int i=0;i<npts;i++) {
			int u = fpts[i];
			int v = fpts[i+npts];

			// get median filtered depth
			vector<float> depths;
			depths.reserve(16);
			for(int nu=0;nu<mfilterSize;nu++) {
				for(int nv=0;nv<mfilterSize;nv++) {
					int idx = ((v+neighbors[nv])*w+(u+neighbors[nu]))*4;		
					float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);
					depths.push_back(d);
				}
			}
			std::sort(depths.begin(), depths.end());

			lms[i].x = u;
			lms[i].y = v;
			lms[i].z = depths[mfilterSize*mfilterSize/2];
		}

		bindTargetLandmarks(lms);
		tOther.toc();

		tRecon.tic();
		if( frameIdx++ == 0 ) {
			vector<unsigned char> colorimg(colordata, colordata+640*480*4);
			vector<unsigned char> depthimg(depthdata, depthdata+640*480*4);
			bindRGBDImage(colorimg, depthimg);
			// fit the pose first, then fit the identity and pose together
			recon.fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
			recon.fitICP(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		}
		else{
			recon.fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
		}
		tRecon.toc();

		tOther.tic();
		pose.assign(recon.getPose(), recon.getPose()+7);
		tOther.toc();

		tTotal.toc();
		return true;
	}
}

void PoseTracker::reset()
{
	aam.reset();
	recon.reset();
}

float PoseTracker::facialFeatureTrackingError() const {
	return aam.getTrackingError();
}

float PoseTracker::poseEstimationError() const
{
	return recon.reconstructionError();
}


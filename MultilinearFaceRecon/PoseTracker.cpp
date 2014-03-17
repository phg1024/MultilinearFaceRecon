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
	reconGPU.setBaseMesh(mesh);

	loadLandmarks();

	lms.resize(landmarks.size());
	labeledLandmarks.resize(landmarks.size());

	frameIdx = 0;
	trackedFrames = 0;

	aam = new AAMWrapper;
}


PoseTracker::~PoseTracker(void)
{
	delete aam;
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
	fpts = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	if( fpts.empty() ) {
		// tracking failed
		cerr << "AAM tracking failed." << endl;
		tTotal.toc();
		return false;
	}
	else {
		// tracking succeeded
		cout << "AAM tracking succeeded." << endl;

		tOther.tic();
		// get the 3D landmarks and feed to recon manager
		const int mfilterSize = 3;
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

			//PhGUtils::debug("u", u, "v", v, "d", lms[i].z);
		}

		tOther.toc();

		
		if( frameIdx++ == 0 ) {
			tSetup.tic();
			vector<unsigned char> colorimg(colordata, colordata+640*480*4);
			vector<unsigned char> depthimg(depthdata, depthdata+640*480*4);
			recon.bindRGBDTarget(colorimg, depthimg);

			for(int i=0;i<landmarks.size();i++) {
				int vidx = landmarks[i];
				labeledLandmarks[i] = (make_pair(lms[i], vidx));
			}
			recon.bindTarget(labeledLandmarks, MultilinearReconstructor::TargetType_2D);

			// fit the pose first, then fit the identity and pose together
			recon.fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);

			// transfer the result to GPU
			// rigid transformation parameters
			reconGPU.setPose( recon.getPose() );
			// identity weights
			reconGPU.setIdentityWeights( recon.identityWeights() );
			// expression weights
			reconGPU.setExpressionWeights( recon.expressionWeights() );

			// fit the pose and identity with GPU ICP
			reconGPU.bindRGBDTarget(colorimg, depthimg);
			reconGPU.bindTarget(lms);			

			reconGPU.fit(MultilinearReconstructorGPU::FIT_POSE_AND_IDENTITY);
			tSetup.toc();
		}
		else{
			tRecon.tic();
			vector<unsigned char> colorimg(colordata, colordata+640*480*4);
			vector<unsigned char> depthimg(depthdata, depthdata+640*480*4);
			reconGPU.bindTarget(lms);
			reconGPU.bindRGBDTarget(colorimg, depthimg);
			reconGPU.fit(MultilinearReconstructorGPU::FIT_POSE_AND_EXPRESSION);
			tRecon.toc();
		}		

		tOther.tic();
		pose.assign(reconGPU.getPose(), reconGPU.getPose()+7);
		tOther.toc();

		tTotal.toc();
		return true;
	}
}

void PoseTracker::reset()
{
	aam->reset();
	recon.reset();
	frameIdx = 0;
	trackedFrames = 0;
}

float PoseTracker::facialFeatureTrackingError() const {
	return aam->getTrackingError();
}

float PoseTracker::poseEstimationError() const
{
	return reconGPU.reconstructionError();
}



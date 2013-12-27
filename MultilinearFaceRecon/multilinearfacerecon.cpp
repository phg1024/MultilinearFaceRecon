#include "multilinearfacerecon.h"
#include <QFileDialog>
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"

MultilinearFaceRecon::MultilinearFaceRecon(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// setup viewer for kinect streams
	setupStreamViews();

	// setup the widget
	viewer = new BlendShapeViewer(this);
	this->setCentralWidget((QWidget*)viewer);

	setupKinectManager();
	lms.resize(128);
	frameIdx = 0;

	connect(ui.actionLoad_Target, SIGNAL(triggered()), this, SLOT(loadTargetMesh()));
	connect(ui.actionFit, SIGNAL(triggered()), this, SLOT(fit()));
	connect(ui.actionGenerate_Prior, SIGNAL(triggered()), this, SLOT(generatePrior()));
	connect(ui.actionStart_Kinect, SIGNAL(triggered()), this, SLOT(toggleKinectInput()));
	connect(ui.actionReset_Tracking, SIGNAL(triggered()), this, SLOT(resetAAM()));
	connect(ui.actionBatch_Recon, SIGNAL(triggered()), this, SLOT(reconstructionWithBatchInput()));

	// timer for kinect input
	connect(&timer, SIGNAL(timeout()), this, SLOT(updateKinectStreams()));

	useKinectInput = false;
}

MultilinearFaceRecon::~MultilinearFaceRecon()
{
	delete viewer;
}

void MultilinearFaceRecon::setupKinectManager() {
	kman.setMode( PhGUtils::KinectManager::WarpDepth );
	
	// 30 frames per second
	timer.setInterval(33);
}

void MultilinearFaceRecon::setupStreamViews() {
	colorView = shared_ptr<StreamViewer>(new StreamViewer());
	colorView->setWindowTitle("Color View");
	depthView = shared_ptr<StreamViewer>(new StreamViewer());
	depthView->setWindowTitle("Depth View");

	colorView->show();
	depthView->show();
}

void MultilinearFaceRecon::loadTargetMesh()
{
	QString filename = QFileDialog::getOpenFileName();
	viewer->bindTargetMesh(filename.toStdString());
}

void MultilinearFaceRecon::fit() {
	viewer->fit();
}

void MultilinearFaceRecon::generatePrior() {
	viewer->generatePrior();
}

int MultilinearFaceRecon::reconstructionWithSingleFrame(
	const unsigned char* colordata,
	const unsigned char* depthdata,
	vector<float>& pose,
	vector<float>& fpts
	) 
{
	const int w = 640, h = 480;


	// AAM tracking
	fpts = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	if( fpts.empty() ) {
		// tracking failed
		cerr << "AAM tracking failed." << endl;
		return -1;
	}
	else {
		// get the 3D landmarks and feed to recon manager
		int npts = fpts.size()/2;
		for(int i=0;i<npts;i++) {
			int u = fpts[i];
			// flip y coordinates
			int v = fpts[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			PhGUtils::colorToWorld(u, v, d, lms[i].x, lms[i].y, lms[i].z);
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);

			// minus one is a hack to bring the model nearer
			//lms[i].z += 1.0;
		}

		viewer->bindTargetLandmarks(lms);
		if( frameIdx++ == 0 ) {
			// fit the pose first, then fit the identity and pose together
			
			viewer->fit(MultilinearReconstructor::FIT_ALL);
		}
		else{
			viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
		}

		const MultilinearReconstructor& recon = viewer->getReconstructor();
		pose.assign(recon.getPose(), recon.getPose()+7);
		/*
		for(int i=0;i<pose.size();i++)
			cout << pose[i] << ((i==pose.size()-1)?'\n':'\t');
		*/
		return 0;
	}
}

void MultilinearFaceRecon::reconstructionWithBatchInput() {
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Fuhao_\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const int startIdx = 10000;
	const int imageCount = 500;
	const int endIdx = startIdx + imageCount;
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";

	const int w = 640;
	const int h = 480;

	PhGUtils::Timer tRecon, tCombined;
	int validFrames = 0;
	frameIdx = 0;

	tCombined.tic();
	for(int imgidx=1;imgidx<=imageCount;imgidx++) {
		// process each image and perform reconstruction
		string colorImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + colorPostfix;
		string depthImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + depthPostfix;
		vector<unsigned char> colordata = PhGUtils::fromQImage(colorImageName);
		vector<unsigned char> depthdata = PhGUtils::fromQImage(depthImageName);

		colorView->bindStreamData(&(colordata[0]), w, h);
		depthView->bindStreamData(&(depthdata[0]), w, h);

		//rgbimg.save("rgb.png");
		//depthimg.save("depth.png");
#if 0
		vector<float> f, pose;
		tRecon.tic();
		reconstructionWithSingleFrame(&(colordata[0]), &(depthdata[0]), pose, f);
		tRecon.toc();
					
		if( f.empty() ) continue;
		validFrames++;

		colorView->bindLandmarks(f);

		QApplication::processEvents();		
		::system("pause");
#else
		vector<float> f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;

		// get the 3D landmarks and feed to recon manager
		int npts = f.size()/2;
		for(int i=0;i<npts;i++) {
			int u = f[i];
			// flip y coordinates
			int v = f[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			PhGUtils::colorToWorld(u, v, d, lms[i].x, lms[i].y, lms[i].z);
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);

			// minus one is a hack to bring the model nearer
			//lms[i].z += 1.0;
		}

		viewer->bindTargetLandmarks(lms);
		if( imgidx == 1 ) {
			// fit the pose first, then fit the identity and pose together
			//viewer->fit(MultilinearReconstructor::FIT_POSE);
			//viewer->fit(MultilinearReconstructor::FIT_IDENTITY);
			//::system("pause");
			//viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
			viewer->fit(MultilinearReconstructor::FIT_ALL);
		}
		else{
			validFrames++;
			tRecon.tic();
			viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
			tRecon.toc();
		}
			//viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
			//viewer->fit(MultilinearReconstructor::FIT_POSE);

		QApplication::processEvents();
		::system("pause");
#endif	
	}
	tCombined.toc();
	PhGUtils::message("Average reconstruction time = " + PhGUtils::toString(tRecon.elapsed() / validFrames));
	PhGUtils::message("Average tracking+recon time = " + PhGUtils::toString(tCombined.elapsed() / validFrames));
}


void MultilinearFaceRecon::updateKinectStreams_2D()
{
	tKman.tic();
	kman.updateStream();
	int w = kman.getWidth(), h = kman.getHeight();
	const vector<unsigned char>& colordata = kman.getRGBData();
	const vector<unsigned char>& depthdata = kman.getDepthData();
	const vector<USHORT>& depthvalues = kman.getDepthValues();
	tKman.toc();

	tView.tic();
	colorView->bindStreamData(&(colordata[0]), w, h);
	depthView->bindStreamData(&(depthdata[0]), w, h);
	tView.toc();
	//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//rgbimg.save("rgb.png");
	//depthimg.save("depth.png");
	tAAM.tic();
	const vector<float>& f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	tView.tic();
	colorView->bindLandmarks(f);
	tView.toc();

	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	tRecon.tic();
	// get the 3D landmarks and feed to recon manager
	int npts = f.size()/2;
	for(int i=0;i<npts;i++) {
		int u = f[i];
		// flip y coordinates
		int v = f[i+npts];
		int idx = (v*w+u)*4;
		float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);
		lms[i].x = u;
		lms[i].y = v;
		lms[i].z = d;
		//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
	}
	viewer->bindTargetLandmarks(lms);
	if( frameIdx++ == 0 ) {
		viewer->fit2d(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
	}
	else
		viewer->fit2d(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
	tRecon.toc();
}


void MultilinearFaceRecon::updateKinectStreams()
{	
	tKman.tic();
	kman.updateStream();
	int w = kman.getWidth(), h = kman.getHeight();
	const vector<unsigned char>& colordata = kman.getRGBData();
	const vector<unsigned char>& depthdata = kman.getDepthData();
	const vector<USHORT>& depthvalues = kman.getDepthValues();
	tKman.toc();

	tView.tic();
	colorView->bindStreamData(&(colordata[0]), w, h);
	depthView->bindStreamData(&(depthdata[0]), w, h);
	tView.toc();
	//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//rgbimg.save("rgb.png");
	//depthimg.save("depth.png");
	tAAM.tic();
	const vector<float>& f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	tView.tic();
	colorView->bindLandmarks(f);
	tView.toc();
	
	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	tRecon.tic();
	// get the 3D landmarks and feed to recon manager
	int npts = f.size()/2;
	for(int i=0;i<npts;i++) {
		int u = f[i];
		// flip y coordinates
		int v = f[i+npts];
		int idx = (v*w+u)*4;
		float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);
		PhGUtils::colorToWorld(u, v, d, lms[i].x, lms[i].y, lms[i].z);
		//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
		
		// minus one is a hack to bring the model nearer
		//lms[i].z += 1.0;
	}
	viewer->bindTargetLandmarks(lms);
	if( frameIdx++ == 0 ) {
		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
	}
	else
		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
	tRecon.toc();
}

void MultilinearFaceRecon::toggleKinectInput()
{
	useKinectInput = !useKinectInput;
	if( useKinectInput ){
		tAAM.reset(); tView.reset(); tKman.reset(); tRecon.reset();
		frameIdx = 0;
		timer.start();
	}
	else{
		timer.stop();
		PhGUtils::message("Total frames = " + PhGUtils::toString(frameIdx));
		PhGUtils::message("Time cost for AAM = " + PhGUtils::toString(tAAM.elapsed() / frameIdx ) + " seconds.");
		PhGUtils::message("Time cost for Kinect = " + PhGUtils::toString(tKman.elapsed() / frameIdx) + " seconds.");
		PhGUtils::message("Time cost for Views = " + PhGUtils::toString(tView.elapsed() / frameIdx) + " seconds.");
		PhGUtils::message("Time cost for Reconstruction = " + PhGUtils::toString(tRecon.elapsed() / frameIdx) + " seconds.");
	}
}

void MultilinearFaceRecon::resetAAM()
{
	timer.stop();
	aam.reset();
	timer.start();
}
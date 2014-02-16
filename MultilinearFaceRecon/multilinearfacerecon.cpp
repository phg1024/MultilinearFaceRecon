#include "multilinearfacerecon.h"
#include <QFileDialog>
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"

#define DOUG 1
#define PEIHONG 0
#define YILONG 0

MultilinearFaceRecon::MultilinearFaceRecon(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// setup viewer for kinect streams
	setupStreamViews();

	// setup the widget
	viewer = new BlendShapeViewer(this);
	this->setCentralWidget((QWidget*)viewer);

	aam = new AAMWrapper;

	setupKinectManager();
	lms.resize(78);
	frameIdx = 0;

	connect(ui.actionLoad_Target, SIGNAL(triggered()), this, SLOT(loadTargetMesh()));
	connect(ui.actionFit, SIGNAL(triggered()), this, SLOT(fit()));
	connect(ui.actionGenerate_Prior, SIGNAL(triggered()), this, SLOT(generatePrior()));
	connect(ui.actionStart_Kinect, SIGNAL(triggered()), this, SLOT(toggleKinectInput()));
	connect(ui.actionStart_Kinect_2D, SIGNAL(triggered()), this, SLOT(toggleKinectInput_2D()));
	connect(ui.actionStart_Kinect_ICP, SIGNAL(triggered()), this, SLOT(toggleKinectInput_ICP()));
	connect(ui.actionReset_Tracking, SIGNAL(triggered()), this, SLOT(resetAAM()));
	connect(ui.actionBatch_Recon, SIGNAL(triggered()), this, SLOT(reconstructionWithBatchInput()));
	connect(ui.actionBatch_Recon_ICP, SIGNAL(triggered()), this,  SLOT(reconstructionWithBatchInput_ICP()));
	connect(ui.actionBatch_Recon_ICP_GPU, SIGNAL(triggered()), this,  SLOT(reconstructionWithBatchInput_GPU()));

	connect(ui.actionRecord, SIGNAL(triggered()), this, SLOT(toggleKinectInput_Record()));

	// timer for kinect input
	connect(&timer, SIGNAL(timeout()), this, SLOT(updateKinectStreams()));
	connect(&timer2d, SIGNAL(timeout()), this, SLOT(updateKinectStreams_2D()));
	connect(&timerICP, SIGNAL(timeout()), this, SLOT(updateKinectStreams_ICP()));
	connect(&timerRecord, SIGNAL(timeout()), this,  SLOT(updateKinectStreams_Record()));

	useKinectInput = false;
}

MultilinearFaceRecon::~MultilinearFaceRecon()
{
	delete viewer;
	delete aam;
}

void MultilinearFaceRecon::setupKinectManager() {
	kman.setMode( PhGUtils::KinectManager::WarpDepth );

	// 30 frames per second
	timer.setInterval(33);

	// 30 frames per second
	timer2d.setInterval(33);

	timerICP.setInterval(33);

	timerRecord.setInterval(33);
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
	fpts = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
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
			int v = fpts[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			PhGUtils::colorToWorld(u, v, d, lms[i].x, lms[i].y, lms[i].z);
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
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
#if DOUG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 250;
#elif PEIHONG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Peihong\\";
	const string imageName = "Peihong_";
	const string colorPostfix = "_color.png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10001;
	const int imageCount = 250;
#else
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#endif

	const int endIdx = startIdx + imageCount;

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
		vector<float> f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;

		// get the 3D landmarks and feed to recon manager
		int npts = f.size()/2;
		for(int i=0;i<npts;i++) {
			int u = f[i];
			int v = f[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			lms[i].x = u;
			lms[i].y = v;
			lms[i].z = d;
		}

		viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);
		if( imgidx == 1 ) {
			// fit the pose first, then fit the identity and pose together
			//viewer->fit(MultilinearReconstructor::FIT_POSE);
			//viewer->fit(MultilinearReconstructor::FIT_IDENTITY);
			//::system("pause");
			viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
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


void MultilinearFaceRecon::reconstructionWithBatchInput_GPU()
{
#if DOUG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 64;
#elif PEIHONG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Peihong\\";
	const string imageName = "Peihong_";
	const string colorPostfix = "_color.png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10001;
	const int imageCount = 90;
#else
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#endif

	const int endIdx = startIdx + imageCount;

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
		
		vector<float> f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;

		// also bind the 3D feature points
		// get the 3D landmarks and feed to recon manager
		int npts = f.size()/2;
		for(int i=0;i<npts;i++) {
			int u = f[i];
			int v = f[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			// pass 2D features plus depth to the recon			
			lms[i].x = u;
			lms[i].y = v;
			lms[i].z = d;
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
		}

		if( imgidx == 1 ) {
			viewer->bindRGBDTarget(colordata, depthdata);
			viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);
			// fit the pose first, then fit the identity and pose together
			viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
			//viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE_AND_IDENTITY);

			viewer->transferParameters(BlendShapeViewer::CPUToGPU);

			viewer->bindRGBDTargetGPU(colordata, depthdata);
			viewer->bindTargetLandmarksGPU(lms);
			viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE_AND_IDENTITY);
		}
		else{
			viewer->bindRGBDTargetGPU(colordata, depthdata);
			viewer->bindTargetLandmarksGPU(lms);
			validFrames++;
			tRecon.tic();
			viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE_AND_EXPRESSION);
			tRecon.toc();
		}

		QApplication::processEvents();
		//::system("pause");
#endif	
	}
	tCombined.toc();
	PhGUtils::message("Average reconstruction time = " + PhGUtils::toString(tRecon.elapsed() / validFrames));
	PhGUtils::message("Average tracking+recon time = " + PhGUtils::toString(tCombined.elapsed() / validFrames));
	viewer->printStatsGPU();
}

void MultilinearFaceRecon::reconstructionWithBatchInput_ICP()
{
#if DOUG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 250;
#elif PEIHONG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Peihong\\";
	const string imageName = "Peihong_";
	const string colorPostfix = "_color.png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10001;
	const int imageCount = 250;
#else
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#endif

	const int endIdx = startIdx + imageCount;

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
		vector<float> f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;

		// transfer the RGBD data to recon
		viewer->bindRGBDTarget(colordata, depthdata);

		// also bind the 3D feature points
		// get the 3D landmarks and feed to recon manager
		int npts = f.size()/2;
		for(int i=0;i<npts;i++) {
			int u = f[i];
			int v = f[i+npts];
			int idx = (v*w+u)*4;
			float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

			// pass 2D features plus depth to the recon			
			lms[i].x = u;
			lms[i].y = v;
			lms[i].z = d;
			//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
		}
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);

		if( imgidx == 1 ) {
			// fit the pose first, then fit the identity and pose together
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			//viewer->fitICP(MultilinearReconstructor::FIT_IDENTITY);
			viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			viewer->fitICP(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		}
		else{
			validFrames++;
			tRecon.tic();
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			viewer->fitICP(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
			tRecon.toc();
		}

		QApplication::processEvents();
		//::system("pause");
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
	const vector<float>& f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
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
		int v = f[i+npts];
		int idx = (v*w+u)*4;
		float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

		lms[i].x = u;
		lms[i].y = v;
		lms[i].z = d;
		//PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
	}
	viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);
	if( frameIdx++ == 0 ) {
		viewer->fit2d(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		//viewer->fit2d(MultilinearReconstructor::FIT_POSE);
	}
	else
		viewer->fit2d(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
	//viewer->fit2d(MultilinearReconstructor::FIT_POSE);
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

	/*
	QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());
	rgbimg.save("rgb.png");
	depthimg.save("depth.png");
	*/

	tAAM.tic();
	const vector<float>& f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	tView.tic();
	colorView->bindLandmarks(f);
	tView.toc();

	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	tRecon.tic();
	// get the 3D landmarks and feed to recon manager
	int mfilterSize = 3;
	int neighbors[] = {-1, 0, 1};

	int npts = f.size()/2;
	for(int i=0;i<npts;i++) {
		int u = f[i];
		int v = f[i+npts];

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
	viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);
	if( frameIdx++ == 0 ) {
		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
	}
	else
		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
	tRecon.toc();
}

void MultilinearFaceRecon::updateKinectStreams_ICP()
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

	/*
	QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());
	rgbimg.save("rgb.png");
	depthimg.save("depth.png");
	*/

	tAAM.tic();
	const vector<float>& f = aam->track(&(colordata[0]), &(depthdata[0]), w, h);
	tAAM.toc();

	tView.tic();
	colorView->bindLandmarks(f);
	tView.toc();

	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	tRecon.tic();
	// get the 3D landmarks and feed to recon manager
	int mfilterSize = 3;
	int neighbors[] = {-1, 0, 1};

	int npts = f.size()/2;
	for(int i=0;i<npts;i++) {
		int u = f[i];
		int v = f[i+npts];

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

	if( frameIdx++ == 0 ) {
		// transfer the RGBD data to recon
		viewer->bindRGBDTarget(colordata, depthdata);

		// also bind the 3D feature points
		// get the 3D landmarks and feed to recon manager
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);

		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		viewer->fitICP(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
	}
	else{
		// also bind the 3D feature points
		// get the 3D landmarks and feed to recon manager
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor::TargetType_2D);

		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
	}
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

void MultilinearFaceRecon::toggleKinectInput_2D()
{
	useKinectInput = !useKinectInput;
	if( useKinectInput ){
		tAAM.reset(); tView.reset(); tKman.reset(); tRecon.reset();
		frameIdx = 0;
		timer2d.start();
	}
	else{
		timer2d.stop();
		PhGUtils::message("Total frames = " + PhGUtils::toString(frameIdx));
		PhGUtils::message("Time cost for AAM = " + PhGUtils::toString(tAAM.elapsed() / frameIdx ) + " seconds.");
		PhGUtils::message("Time cost for Kinect = " + PhGUtils::toString(tKman.elapsed() / frameIdx) + " seconds.");
		PhGUtils::message("Time cost for Views = " + PhGUtils::toString(tView.elapsed() / frameIdx) + " seconds.");
		PhGUtils::message("Time cost for Reconstruction = " + PhGUtils::toString(tRecon.elapsed() / frameIdx) + " seconds.");
	}
}

void MultilinearFaceRecon::toggleKinectInput_ICP() {
	useKinectInput = !useKinectInput;
	if( useKinectInput ){
		tAAM.reset(); tView.reset(); tKman.reset(); tRecon.reset();
		frameIdx = 0;
		timerICP.start();
	}
	else{
		timerICP.stop();
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
	aam->reset();
	timer.start();
}

void MultilinearFaceRecon::updateKinectStreams_Record()
{
	kman.updateStream();
	int w = kman.getWidth(), h = kman.getHeight();
	vector<unsigned char> colordata = kman.getRGBData();
	vector<unsigned char> depthdata = kman.getDepthData();

	colorView->bindStreamData(&(colordata[0]), w, h);
	depthView->bindStreamData(&(depthdata[0]), w, h);

	recordData.push_back(make_pair(colordata, depthdata));
}

void MultilinearFaceRecon::toggleKinectInput_Record()
{
	useKinectInput = !useKinectInput;
	if( useKinectInput ){
		frameIdx = 0;
		timerRecord.start();

		recordData.clear();
		recordData.reserve(1200);
	}
	else{
		timerRecord.stop();

		PhGUtils::message("Wring out recorded images ...");
		// write recorded data
		const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Peihong\\";
		const string imageName = "Peihong_";
		const int startIdx = 10000;
		const int imageCount = recordData.size();
		const int endIdx = startIdx + imageCount;
		const string colorPostfix = "_color.png";
		const string depthPostfix = "_depth.png";

		const int w = 640;
		const int h = 480;

		for(int i=0;i<imageCount;i++) {
			// process each image and perform reconstruction
			PhGUtils::message("Processing frame #" + PhGUtils::toString(i) + " out of " + PhGUtils::toString(imageCount) + " frames ...");
			string colorImageName = path + imageName + PhGUtils::toString(startIdx+i) + colorPostfix;
			string depthImageName = path + imageName + PhGUtils::toString(startIdx+i) + depthPostfix;

			QImage cimg = PhGUtils::toQImage(&(recordData[i].first[0]), w, h);
			cimg.save(colorImageName.c_str());
			QImage dimg = PhGUtils::toQImage(&(recordData[i].second[0]), w, h);
			dimg.save(depthImageName.c_str());
		}
		PhGUtils::message("done.");
	}
}

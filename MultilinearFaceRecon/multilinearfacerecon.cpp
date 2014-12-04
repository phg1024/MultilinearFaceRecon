#include "multilinearfacerecon.h"
#include <QFileDialog>
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"

#define DOUG 0
#define PEIHONG 1
#define YILONG 0
#define YILONG2 0

MultilinearFaceRecon::MultilinearFaceRecon(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// setup viewer for kinect streams
	setupStreamViews();

	// setup the widget
	viewer = new BlendShapeViewer(this);
	this->setCentralWidget((QWidget*)viewer);

	// esr tracker
  tracker.reset(new tracker_t());
  tracker->setImageWidth(640);
  tracker->setImageHeight(480);

  // plain point file tracker
  pointfile_tracker.reset(new pointfile_tracker_t());

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
  connect(ui.actionSingle_Recon, SIGNAL(triggered()), this, SLOT(reconstructionWithSingleImage()));
  
  connect(ui.actionSingle_Image, SIGNAL(triggered()), this, SLOT(reconstructionWithSettings()));
  connect(ui.actionMultiple_Images, SIGNAL(triggered()), this, SLOT(reconstructionWithMultipleImages()));
  
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
	//delete aam;
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
#if 0
	const int w = 640, h = 480;


	// AAM tracking
  fpts = tracker->track(&(colordata[0]), &(depthdata[0]));
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

			viewer->fit(MultilinearReconstructor_old::FIT_ALL);
		}
		else{
			viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
		}

		const MultilinearReconstructor_old& recon = viewer->getReconstructor();
		pose.assign(recon.getPose(), recon.getPose()+7);
		/*
		for(int i=0;i<pose.size();i++)
		cout << pose[i] << ((i==pose.size()-1)?'\n':'\t');
		*/
		return 0;
	}
#else

  return 0;
#endif
}

void MultilinearFaceRecon::reconstructionWithBatchInput() {
#if DOUG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 64;
#elif PEIHONG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Peihong\\";
	const string imageName = "Peihong_";
	const string colorPostfix = "_color.png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 250;

#elif YILONG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#elif YILONG2
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test\\";
	const string imageName = "000";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 590;
	const int imageCount = 181;
#endif

	const int endIdx = startIdx + imageCount;

	const int w = 640;
	const int h = 480;

	PhGUtils::Timer tRecon, tCombined;
	int validFrames = 0;
	frameIdx = 0;
  	
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
    tCombined.tic();
    vector<float> f = tracker->track(&(colordata[0]), &(depthdata[0]));
    tCombined.toc();
    tracker->printTimeStats();
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;
    tCombined.tic();
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

		viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
		if( imgidx == 1 ) {
			// fit the pose first, then fit the identity and pose together
			//viewer->fit(MultilinearReconstructor::FIT_POSE);
			//viewer->fit(MultilinearReconstructor::FIT_IDENTITY);
			//::system("pause");
      viewer->fit2d(MultilinearReconstructor_old::FIT_POSE);
      QApplication::processEvents();
      ::system("pause");
      viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
      //viewer->fit2d(MultilinearReconstructor::FIT_POSE);
      QApplication::processEvents();
      ::system("pause");
		}
		else{
			validFrames++;
			tRecon.tic();
      //viewer->fit2d(MultilinearReconstructor::FIT_POSE);
      viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
			tRecon.tocMS("Reconstruction");
		}
		//viewer->fit(MultilinearReconstructor::FIT_POSE_AND_IDENTITY);
		//viewer->fit(MultilinearReconstructor::FIT_POSE);
    tCombined.toc();

		QApplication::processEvents();
		::system("pause");
#endif	
	}
	PhGUtils::message("Average reconstruction time = " + PhGUtils::toString(tRecon.elapsed() * 1000.0 / validFrames) + "ms");
	PhGUtils::message("Average tracking+recon time = " + PhGUtils::toString(tCombined.elapsed() * 1000.0 / validFrames) + "ms");
}

#if USE_GPU_RECON
void MultilinearFaceRecon::reconstructionWithBatchInput_GPU()
{
#if DOUG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 250;
#elif PEIHONG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Peihong\\";
	const string imageName = "Peihong_";
	const string colorPostfix = "_color.png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 10000;
	const int imageCount = 445;
#elif YILONG
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#elif YILONG2
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test\\";
	const string imageName = "000";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 590;
	const int imageCount = 180;
#endif

	const int endIdx = startIdx + imageCount;

	const int w = 640;
	const int h = 480;

	PhGUtils::Timer tRecon, tCombined;
	int validFrames = 0;
	frameIdx = 0;

	tCombined.tic();
	for(int imgidx=0;imgidx<=imageCount;imgidx++) {
		PhGUtils::message("Processing frame " + PhGUtils::toString(imgidx) + "...");
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

#define TRACK_ONLY 0
#if TRACK_ONLY
    vector<float> f = tracker->track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		const string fpPostFix = ".txt";
		// write the tracked feature points to files and 
		string fpFileName = path + imageName + PhGUtils::toString(startIdx+imgidx) + fpPostFix;
		ofstream fpfile(fpFileName);
		fpfile << f.size() / 2 << endl;
		int stride = f.size()/2;
		for(int i=0;i<f.size()/2;i++) {
			fpfile << f[i] << ' ' << f[i+stride] << endl;
		}
		fpfile.close();
		continue;
#else
#define USE_REALTIME_TRACKING 1
#if USE_REALTIME_TRACKING
    vector<float> f = tracker->track(&(colordata[0]), &(depthdata[0]));
		colorView->bindLandmarks(f);
#else
		vector<float> f;

		const string fpPostFix = ".txt";
		// write the tracked feature points to files and 
		string fpFileName = path + imageName + PhGUtils::toString(startIdx+imgidx) + fpPostFix;
		ifstream fpfile(fpFileName);

		// load feature points from file
		int npoints;
		fpfile >> npoints;
		f.resize(npoints*2);

		for(int i=0;i<npoints;i++) {
			fpfile >> f[i] >> f[i+npoints];
		}

		fpfile.close();

		colorView->bindLandmarks(f);
#endif
#endif


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

			// get median filtered depth
			const int mfilterSize = 3;
			int neighbors[] = {-1, 0, 1};
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

		if( imgidx == 0 ) {
			viewer->bindRGBDTarget(colordata, depthdata);
			viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
			// fit the pose first, then fit the identity and pose together
			viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
			//viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE_AND_IDENTITY);

			viewer->transferParameters(BlendShapeViewer::CPUToGPU);

			viewer->bindRGBDTargetGPU(colordata, depthdata);
			viewer->bindTargetLandmarksGPU(lms);
			viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE_AND_IDENTITY);
			//viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE);
		}
		else{
			viewer->bindRGBDTargetGPU(colordata, depthdata);
			viewer->bindTargetLandmarksGPU(lms);
			validFrames++;
			tRecon.tic();
			//viewer->fitICP_GPU(MultilinearReconstructorGPU::FIT_POSE);
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
#else
void MultilinearFaceRecon::reconstructionWithBatchInput_GPU() {
}
#endif

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
#elif YILONG
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 2670;
	const int imageCount = 120;
#elif YILONG2
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test\\";
	const string imageName = "000";
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
	const int startIdx = 590;
	const int imageCount = 181;
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
    vector<float> f = tracker->track(&(colordata[0]), &(depthdata[0]));
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
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);

		if( imgidx == 1 ) {
			// fit the pose first, then fit the identity and pose together
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			//viewer->fitICP(MultilinearReconstructor::FIT_IDENTITY);
			viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			viewer->fitICP(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
		}
		else{
			validFrames++;
			tRecon.tic();
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);
			//viewer->fitICP(MultilinearReconstructor::FIT_POSE);
			viewer->fitICP(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
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
  const vector<float>& f = tracker->track(&(colordata[0]), &(depthdata[0]));
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
	viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
	if( frameIdx++ == 0 ) {
		viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
		//viewer->fit2d(MultilinearReconstructor::FIT_POSE);
	}
	else
		viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
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
  const vector<float>& f = tracker->track(&(colordata[0]), &(depthdata[0]));
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
	viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
	if( frameIdx++ == 0 ) {
		viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
	}
	else
		viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
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
  const vector<float>& f = tracker->track(&(colordata[0]), &(depthdata[0]));
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
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);

		viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
		viewer->fitICP(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
	}
	else{
		// also bind the 3D feature points
		// get the 3D landmarks and feed to recon manager
		viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);

		viewer->fit(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
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
  tracker->reset();
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
		const string path = "C:\\Users\\PhG\\Desktop\\Data\\Peihong\\";
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

void MultilinearFaceRecon::reconstructionWithSingleImage()
{
  // load an selected image
  QString filename = QFileDialog::getOpenFileName();
  QImage inimg(filename);
  cout << "image size: " << inimg.width() << "x" << inimg.height() << endl;

  // scale down the image if necessary
  int w = inimg.width(), h = inimg.height();
  int longside = max(w, h);
  float factor = 640 / (float)longside;
  inimg = inimg.scaled(w*factor, h*factor);
  cout << "new image size: " << inimg.width() << "x" << inimg.height() << endl;
  w = inimg.width();
  h = inimg.height();

  // resize the viewer accordingly
  //viewer->resize(w, h);
  colorView->resize(w, h);
  this->resize(w, h + 53);
  cout << "canvas size: " << viewer->width() << "x" << viewer->height() << endl;
  // obtain the data
  vector<unsigned char> colordata = PhGUtils::fromQImage(inimg);

  colorView->bindStreamData(&colordata[0], w, h);
  depthView->hide();

  // update the tracker
  int w0 = tracker->getImageWidth();
  int h0 = tracker->getImageHeight();

  tracker->setImageSize(w, h);

  cout << "finding feature points..." << endl;
  vector<float> fpts = tracker->track(&colordata[0], NULL);
  tracker->printTimeStats();
  colorView->bindLandmarks(fpts);

  // reconstruction
  if (fpts.empty()) return;
  
  // get the 3D landmarks and feed to recon manager
  int npts = fpts.size() / 2;
  for (int i = 0; i < npts; i++) {
    int u = fpts[i];
    int v = fpts[i + npts];
    int idx = (v*w + u) * 4;
    float d = 0.99995;

    lms[i].x = u;
    lms[i].y = v;
    lms[i].z = d;
  }

  viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
  viewer->setReconstructionImageSize(w, h);
  viewer->bindImage(colordata);
  viewer->resetReconstructor();
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE);
  QApplication::processEvents();
  ::system("pause");
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
  QApplication::processEvents();
  ::system("pause");
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
}

void MultilinearFaceRecon::reconstructionWithSettings()
{
  /// load the setting file
  QString filename = QFileDialog::getOpenFileName();
  ifstream settings(filename.toStdString());
  string imgfile, ptsfile, lmsfile;
  settings >> imgfile >> ptsfile >> lmsfile;
  settings.close();

  cout << imgfile << "\n" << ptsfile << "\n" << lmsfile << endl;

  /// set the landmark indices for the viewer
  viewer->loadLandmarks(lmsfile);

  QImage inimg(QString(imgfile.c_str()));
  cout << "image size: " << inimg.width() << "x" << inimg.height() << endl;

  // scale down the image if necessary
  int w = inimg.width(), h = inimg.height();
  int longside = max(w, h);
  float factor = 640 / (float)longside;
  inimg = inimg.scaled(w*factor, h*factor);
  cout << "new image size: " << inimg.width() << "x" << inimg.height() << endl;
  w = inimg.width();
  h = inimg.height();

  // resize the viewer accordingly
  //viewer->resize(w, h);
  colorView->resize(w, h);
  this->resize(w, h + 53);
  cout << "canvas size: " << viewer->width() << "x" << viewer->height() << endl;
  // obtain the data
  vector<unsigned char> colordata = PhGUtils::fromQImage(inimg);

  colorView->bindStreamData(&colordata[0], w, h);
  depthView->hide();

  // update the tracker
  cout << "finding feature points..." << endl;
  pointfile_tracker->setGroudTruthFile(ptsfile);
  vector<float> fpts = pointfile_tracker->track(NULL, NULL);
  pointfile_tracker->printTimeStats();

  // process the feature points
  std::for_each(fpts.begin(), fpts.end(), [=](float &x){ x *= factor; });

  colorView->bindLandmarks(fpts);
  cout << "binded." << endl;
  // reconstruction
  if (fpts.empty()) return;

  // get the 3D landmarks and feed to recon manager  
  int npts = fpts.size() / 2;
  cout << npts << endl;
  lms.resize(npts);
  for (int i = 0; i < npts; i++) {
    int u = fpts[i];
    int v = fpts[i + npts];
    int idx = (v*w + u) * 4;
    float d = 0.99995;

    lms[i].x = u;
    lms[i].y = v;
    lms[i].z = d;
  }

  viewer->bindTargetLandmarks(lms, MultilinearReconstructor_old::TargetType_2D);
  cout << "binded" << endl;
  viewer->setReconstructionImageSize(w, h);
  cout << "set." << endl;
  viewer->bindImage(colordata);
  cout << "binded." << endl;
  viewer->resetReconstructor();
  cout << "recon reset." << endl;
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE);
  QApplication::processEvents();
  ::system("pause");
  
#if 0
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_IDENTITY);
  QApplication::processEvents();
  ::system("pause");
  viewer->fit2d(MultilinearReconstructor_old::FIT_POSE_AND_EXPRESSION);
  QApplication::processEvents();
  ::system("pause");
  viewer->fit2d(MultilinearReconstructor_old::FIT_ALL);
#else
  viewer->fit2d(MultilinearReconstructor_old::FIT_ALL_PROGRESSIVE);
#endif
}

void MultilinearFaceRecon::reconstructionWithMultipleImages()
{
  /// load the setting file
  QString filename = QFileDialog::getOpenFileName();

  /// get the path
  QFileInfo fileinfo(filename);
  string filepath = fileinfo.path().toStdString();
  cout << filepath << endl;

  ifstream settings(filename.toStdString());
  string imgfile, ptsfile, lmsfile;
  settings >> lmsfile;
  vector<string> imgfiles, ptsfiles;
  while (settings.good()) {
    settings >> imgfile >> ptsfile;
    imgfiles.push_back(filepath + "/" + imgfile);
    ptsfiles.push_back(filepath + "/" + ptsfile);
  }
  settings.close();

  cout << lmsfile << endl;
  ifstream fin(lmsfile, ios::in);
  vector<int> initLandmarks;
  if (fin.is_open()) {
    initLandmarks.reserve(128);
    initLandmarks.clear();
    int idx;
    while (fin.good()) {
      fin >> idx;
      cout << idx << endl;
      initLandmarks.push_back(idx);
    }
    PhGUtils::message("landmarks loaded.");
    cout << "total landmarks = " << initLandmarks.size() << endl;
  }
  else {
    cerr << "Failed to open landmark file. Abort." << endl;
    return;
  }

  for (auto fi : imgfiles) {
    cout << fi << endl;
  }
  for (auto fi : ptsfiles) {
    cout << fi << endl;
  }

  // collect the images
  vector<QImage> imgset;
  vector<PhGUtils::Point2i> imgsizes;
  vector<double> scales;
  for (int i = 0; i < imgfiles.size(); ++i) {
    QImage inimg(QString(imgfiles[i].c_str()));
    cout << "image size: " << inimg.width() << "x" << inimg.height() << endl;

    // resize the image
    int w = inimg.width(), h = inimg.height();
    int longside = max(w, h);
    float factor = 640 / (float)longside;
    inimg = inimg.scaled(w*factor, h*factor);

    imgset.push_back(inimg);
    imgsizes.push_back(PhGUtils::Point2i(inimg.width(), inimg.height()));
    scales.push_back(factor);
  }

  // collect the set of input 2D points
  vector<vector<float>> infpts;
  vector<vector<Constraint_2D>> fptset;
  for (int i = 0; i < ptsfiles.size(); ++i) {
    cout << "finding feature points from " << ptsfiles[i] << endl;
    pointfile_tracker->setGroudTruthFile(ptsfiles[i]);
    auto fpts = pointfile_tracker->track(NULL, NULL);
    infpts.push_back(fpts);
    // reconstruction
    if (fpts.empty()){
      cerr << "Failed to find feature points for image " << imgfiles[i] << endl;
      return;
    }
    else {
      int npts = fpts.size() / 2;
      vector<Constraint_2D> cons(npts);
      for (int j = 0; j < npts; ++j) {
        int u = fpts[j];
        int v = fpts[j + npts];       
        cons[j].q.x = u * scales[i];
        cons[j].q.y = v * scales[i];
        cons[j].weight = 1.0;
        cons[j].vidx = initLandmarks[j];
      }
      fptset.push_back(cons);
    }
  }
  
  cout << "data loaded." << endl;

#if 1
  MultilinearReconstructor<MultilinearModel<double>, Constraint_2D, 
    Optimizer<MultilinearModel<double>, Constraint_2D, MultiImageParameters, MultiImageEngergyFunction2D<double>>, MultiImageParameters> mrecon;

  mrecon.reset(imgfiles.size());
  mrecon.setImageSize(imgsizes);
  mrecon.fit(fptset, FIT_POSE);
  auto results = mrecon.fit(fptset, FittingOptions(FIT_ALL_PROGRESSIVE, 4 * imgfile.size()));

  // write results to files
  viewers.clear();
  for (int i = 0; i < results.size(); ++i) {
    stringstream ss;
    ss << i << ".mesh";
    results[i].write(ss.str());

    viewers.push_back(shared_ptr<MeshViewer>(new MeshViewer));
    auto &vr = viewers.back();
    vr->setWindowTitle(ss.str().c_str());
    cout << vr << endl;
    vr->show();
    vr->bindImage(imgset[i]);
    vr->bindLandmarks(infpts[i]);
    vr->setFocalLength(mrecon.getFocalLength(i));
    float x, y, z;
    mrecon.getTranslation(i, x, y, z);
    vr->setTranslation(x, y, z);
    vr->bindMesh(mrecon.getMesh(i));
    vr->show();
  }
#endif
}

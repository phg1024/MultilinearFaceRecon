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

void MultilinearFaceRecon::reconstructionWithBatchInput() {
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const int startIdx = 10000;
	const int imageCount = 200;
	const int endIdx = startIdx + imageCount;
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";

	const int w = 640;
	const int h = 480;

	for(int imgidx=1;imgidx<imageCount;imgidx++) {
		// process each image and perform reconstruction
		string colorImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + colorPostfix;
		string depthImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + depthPostfix;
		vector<unsigned char> colordata = PhGUtils::fromQImage(colorImageName);
		vector<unsigned char> depthdata = PhGUtils::fromQImage(depthImageName);

		colorView->bindStreamData(&(colordata[0]), w, h);
		depthView->bindStreamData(&(depthdata[0]), w, h);

		//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
		//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

		//rgbimg.save("rgb.png");
		//depthimg.save("depth.png");

		vector<float> f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
		colorView->bindLandmarks(f);

		// do not update the mesh if the landmarks are unknown
		if( f.empty() ) continue;

		// get the 3D landmarks and feed to recon manager
		int npts = f.size()/2;
		vector<PhGUtils::Point3f> lms(npts);
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
		if( imgidx == 0 )
			viewer->fit(MultilinearReconstructor::FIT_IDENTITY);
		else
			viewer->fit(MultilinearReconstructor::FIT_POSE);

		QApplication::processEvents();
		::system("pause");
	}
}

void MultilinearFaceRecon::updateKinectStreams()
{	
	PhGUtils::Timer t, t2;
	t2.tic();
	t.tic();
	kman.updateStream();
	int w = kman.getWidth(), h = kman.getHeight();
	const vector<unsigned char>& colordata = kman.getRGBData();
	const vector<unsigned char>& depthdata = kman.getDepthData();
	const vector<USHORT>& depthvalues = kman.getDepthValues();

	colorView->bindStreamData(&(colordata[0]), w, h);
	depthView->bindStreamData(&(depthdata[0]), w, h);
	t.toc("Update views");
	//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//rgbimg.save("rgb.png");
	//depthimg.save("depth.png");
	t.tic();
	vector<float> f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	t.toc("AAM");
	t.tic();
	colorView->bindLandmarks(f);
	

	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	// get the 3D landmarks and feed to recon manager
	int npts = f.size()/2;
	vector<PhGUtils::Point3f> lms(npts);
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
	if( frameIdx++ == 0 )
		viewer->fit(MultilinearReconstructor::FIT_IDENTITY);
	else
		viewer->fit(MultilinearReconstructor::FIT_POSE_AND_EXPRESSION);

	t.toc("Reconstruction total");
	t2.toc();
	PhGUtils::message("Frame rate = " + PhGUtils::toString(1.0 / t2.gap()) + " fps");
}

void MultilinearFaceRecon::toggleKinectInput()
{
	useKinectInput = !useKinectInput;
	frameIdx = 0;
	if( useKinectInput ) timer.start();
	else timer.stop();
}

void MultilinearFaceRecon::resetAAM()
{
	timer.stop();
	aam.reset();
	timer.start();
}

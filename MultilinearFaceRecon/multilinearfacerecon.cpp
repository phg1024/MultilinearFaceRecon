#include "multilinearfacerecon.h"
#include <QFileDialog>
#include "Utils/utility.hpp"
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
	
	// 25 frames per second
	timer.setInterval(40);
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

void MultilinearFaceRecon::updateKinectStreams()
{	
	kman.updateStream();
	int w = kman.getWidth(), h = kman.getHeight();
	const vector<unsigned char>& colordata = kman.getRGBData();
	const vector<unsigned char>& depthdata = kman.getDepthData();

	colorView->bindStreamData(&(colordata[0]), w, h);
	depthView->bindStreamData(&(depthdata[0]), w, h);

	//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//rgbimg.save("rgb.png");
	//depthimg.save("depth.png");

	vector<float> f = aam.track(&(colordata[0]), &(depthdata[0]), w, h);
	colorView->bindLandmarks(f);

	// do not update the mesh if the landmarks are unknown
	if( f.empty() ) return;

	// get the 3D landmarks and feed to recon manager
	int npts = f.size()/2;
	vector<PhGUtils::Point3f> lms(npts);
	for(int i=0;i<npts;i++) {
		int x = f[i];
		// flip it vertically
		int y = f[i+npts];
		int idx = (y*w+x)*4;
		float z = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

		float X, Y, Z;
		PhGUtils::depthToWorld(w - 1 - x, h - 1 - y, z, X, Y, Z);
		//PhGUtils::debug("x", x, "y", y, "z", z);
		//PhGUtils::debug("X", X, "Y", Y, "Z", Z);
		lms[i] = PhGUtils::Point3f(X, Y, Z);
	}
	viewer->bindTargetLandmarks(lms);
	viewer->fit();
}

void MultilinearFaceRecon::toggleKinectInput()
{
	useKinectInput = !useKinectInput;

	if( useKinectInput ) timer.start();
	else timer.stop();
}

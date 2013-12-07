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
	connect(ui.actionReset_Tracking, SIGNAL(triggered()), this, SLOT(resetAAM()));

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
	const vector<USHORT>& depthvalues = kman.getDepthValues();

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
		int u = f[i];
		// flip y coordinates
		int v = h - 1 - f[i+npts];
		int idx = (v*w+u)*4;
		float d = (depthdata[idx]<<16|depthdata[idx+1]<<8|depthdata[idx+2]);

		float X, Y, Z;
		PhGUtils::colorToWorld(u, v, d, X, Y, Z);
		PhGUtils::debug("u", u, "v", v, "d", d, "X", X, "Y", Y, "Z", Z);
		
		lms[i] = PhGUtils::Point3f(X, Y, Z);
		
		// minus one is a hack to bring the model nearer
		//lms[i].z += 1.0;
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

void MultilinearFaceRecon::resetAAM()
{
	timer.stop();
	aam.reset();
	timer.start();
}

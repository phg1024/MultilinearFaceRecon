#include "multilinearfacerecon.h"
#include <QFileDialog>
#include "Utils/utility.hpp"

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

	connect(&timer, SIGNAL(timeout()), this, SLOT(updateKinectStreams()));

	timer.start();
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
	colorView->bindStreamData(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	depthView->bindStreamData(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//QImage rgbimg = PhGUtils::toQImage(&(kman.getRGBData()[0]), kman.getWidth(), kman.getHeight());
	//QImage depthimg = PhGUtils::toQImage(&(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());

	//rgbimg.save("rgb.png");
	//depthimg.save("depth.png");

	vector<float> f = aam.track(&(kman.getRGBData()[0]), &(kman.getDepthData()[0]), kman.getWidth(), kman.getHeight());
	colorView->bindLandmarks(f);
}

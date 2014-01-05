#ifndef MULTILINEARFACERECON_H
#define MULTILINEARFACERECON_H

#include <QtWidgets/QMainWindow>
#include "ui_multilinearfacerecon.h"

#include <QTimer>
#include "Kinect/KinectManager.h"
#include "Kinect/StreamViewer.h"

#include "BlendShapeViewer.h"
#include "AAMWrapper.h"

#include "Utils/Timer.h"

class MultilinearFaceRecon : public QMainWindow
{
	Q_OBJECT

public:
	MultilinearFaceRecon(QWidget *parent = 0);
	~MultilinearFaceRecon();

public slots:
	void toggleKinectInput();
	void toggleKinectInput_2D();
	void resetAAM();
	void reconstructionWithBatchInput();
	int reconstructionWithSingleFrame(
		const unsigned char* colordata,
		const unsigned char* depthdata,
		vector<float>& pose,
		vector<float>& fpts
	);

	void reconstructionWithBatchInput_ICP();

private:
	void setupStreamViews();
	void setupKinectManager();

private slots:
	void loadTargetMesh();
	void fit();
	void generatePrior();
	void updateKinectStreams();
	void updateKinectStreams_2D();

private:
	Ui::MultilinearFaceReconClass ui;

private:
	QTimer timer, timer2d;
	BlendShapeViewer* viewer;

	// kinect input related
	bool useKinectInput;
	PhGUtils::KinectManager kman;
	int frameIdx;
	vector<PhGUtils::Point3f> lms;		// landmarks got from AAM tracking
	AAMWrapper aam;
	shared_ptr<StreamViewer> colorView, depthView;

	PhGUtils::Timer tAAM, tKman, tView, tRecon;
};

#endif // MULTILINEARFACERECON_H

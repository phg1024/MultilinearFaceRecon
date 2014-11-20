#ifndef MULTILINEARFACERECON_H
#define MULTILINEARFACERECON_H

#include <QtWidgets/QMainWindow>
#include "ui_multilinearfacerecon.h"

#include <QTimer>
#include "Kinect/KinectManager.h"
#include "Kinect/StreamViewer.h"

#include "BlendShapeViewer.h"
#include "FaceAligner.h"
#include "AAMWrapper.h"
#include "ESRWrapper.h"
#include "PointFileTracker.h"

#include "Utils/Timer.h"

#define USE_AAM 0

class MultilinearFaceRecon : public QMainWindow
{
	Q_OBJECT

public:
	MultilinearFaceRecon(QWidget *parent = 0);
	~MultilinearFaceRecon();

public slots:
	void toggleKinectInput();
	void toggleKinectInput_2D();
	void toggleKinectInput_ICP();
	void toggleKinectInput_Record();
	void resetAAM();
	void reconstructionWithBatchInput();
  void reconstructionWithSingleImage();
  void reconstructionWithSettings();
	int reconstructionWithSingleFrame(
		const unsigned char* colordata,
		const unsigned char* depthdata,
		vector<float>& pose,
		vector<float>& fpts
	);

	void reconstructionWithBatchInput_ICP();
	void reconstructionWithBatchInput_GPU();

private:
	void setupStreamViews();
	void setupKinectManager();

private slots:
	void loadTargetMesh();
	void fit();
	void generatePrior();
	void updateKinectStreams();
	void updateKinectStreams_2D();
	void updateKinectStreams_ICP();
	void updateKinectStreams_Record();
	
private:
	Ui::MultilinearFaceReconClass ui;

private:
	QTimer timer, timer2d, timerICP, timerRecord;
	BlendShapeViewer* viewer;

	vector<pair<vector<unsigned char>, vector<unsigned char>>> recordData;

	// kinect input related
	bool useKinectInput;
	PhGUtils::KinectManager kman;
	int frameIdx;
	vector<PhGUtils::Point3f> lms;		// landmarks got from AAM tracking
  typedef FaceAligner<ESRWrapper> tracker_t;
  unique_ptr<tracker_t> tracker;
  typedef FaceAligner<PointFileTracker> pointfile_tracker_t;
  unique_ptr<pointfile_tracker_t> pointfile_tracker;
	shared_ptr<StreamViewer> colorView, depthView;

	PhGUtils::Timer tAAM, tKman, tView, tRecon;
};

#endif // MULTILINEARFACERECON_H

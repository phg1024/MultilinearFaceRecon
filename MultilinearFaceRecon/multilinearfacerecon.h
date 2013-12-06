#ifndef MULTILINEARFACERECON_H
#define MULTILINEARFACERECON_H

#include <QtWidgets/QMainWindow>
#include "ui_multilinearfacerecon.h"

#include <QTimer>
#include "Kinect/KinectManager.h"
#include "Kinect/StreamViewer.h"

#include "BlendShapeViewer.h"
#include "AAMWrapper.h"

class MultilinearFaceRecon : public QMainWindow
{
	Q_OBJECT

public:
	MultilinearFaceRecon(QWidget *parent = 0);
	~MultilinearFaceRecon();

private:
	void setupStreamViews();
	void setupKinectManager();

private slots:
	void loadTargetMesh();
	void fit();
	void generatePrior();
	void updateKinectStreams();

private:
	Ui::MultilinearFaceReconClass ui;

private:
	QTimer timer;
	PhGUtils::KinectManager kman;
	AAMWrapper aam;
	BlendShapeViewer* viewer;
	shared_ptr<StreamViewer> colorView, depthView;
};

#endif // MULTILINEARFACERECON_H

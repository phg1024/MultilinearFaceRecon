#include "multilinearfacerecon.h"
#include <QFileDialog>

MultilinearFaceRecon::MultilinearFaceRecon(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// setup the widget
	viewer = new BlendShapeViewer(this);

	this->setCentralWidget((QWidget*)viewer);

	connect(ui.actionLoad_Target, SIGNAL(triggered()), this, SLOT(loadTargetMesh()));
	connect(ui.actionFit, SIGNAL(triggered()), this, SLOT(fit()));
}

MultilinearFaceRecon::~MultilinearFaceRecon()
{

}

void MultilinearFaceRecon::loadTargetMesh()
{
	QString filename = QFileDialog::getOpenFileName();
	viewer->bindTargetMesh(filename.toStdString());
}

void MultilinearFaceRecon::fit() {
	viewer->fit();
}

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
}

MultilinearFaceRecon::~MultilinearFaceRecon()
{

}

void MultilinearFaceRecon::loadTargetMesh()
{
	QString filename = QFileDialog::getOpenFileName();
	viewer->bindTargetMesh(filename.toStdString());
}

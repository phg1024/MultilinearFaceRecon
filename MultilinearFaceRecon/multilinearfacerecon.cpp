#include "multilinearfacerecon.h"

MultilinearFaceRecon::MultilinearFaceRecon(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	// setup the widget
	viewer = new BlendShapeViewer(this);

	this->setCentralWidget((QWidget*)viewer);
}

MultilinearFaceRecon::~MultilinearFaceRecon()
{

}

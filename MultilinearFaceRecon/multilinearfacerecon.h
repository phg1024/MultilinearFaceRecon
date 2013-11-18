#ifndef MULTILINEARFACERECON_H
#define MULTILINEARFACERECON_H

#include <QtWidgets/QMainWindow>
#include "ui_multilinearfacerecon.h"

#include "BlendShapeViewer.h"

class MultilinearFaceRecon : public QMainWindow
{
	Q_OBJECT

public:
	MultilinearFaceRecon(QWidget *parent = 0);
	~MultilinearFaceRecon();

private:
	Ui::MultilinearFaceReconClass ui;

private:
	BlendShapeViewer* viewer;
};

#endif // MULTILINEARFACERECON_H

#include "multilinearfacerecon.h"
#include <QtWidgets/QApplication>

#include "phgutils.h"
#include "Utils/console.h"

int main(int argc, char *argv[])
{
	createConsole();
	QApplication a(argc, argv);

	glutInit(&argc, argv);

	MultilinearFaceRecon w;
	w.show();
	return a.exec();
}

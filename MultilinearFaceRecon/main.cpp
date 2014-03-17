#include "multilinearfacerecon.h"
#include <QtWidgets/QApplication>

#include "phgutils.h"
#include "Utils/console.h"
#include "Utils/utility.hpp"
#include "Utils/stringutils.h"
#include "Utils/Timer.h"

#include "PoseTracker.h"

void testPoseTracker() {
	PoseTracker ptracker;

#if 1
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\Fuhao\\images\\";
	const string imageName = "DougTalkingComplete_KSeq_";
	const int startIdx = 10000;
	const int imageCount = 250;
	const int endIdx = startIdx + imageCount;
	const string colorPostfix = ".jpg";
	const string depthPostfix = "_depth.png";
#else
	const string path = "C:\\Users\\Peihong\\Desktop\\Data\\Yilong\\test_images\\";
	const string imageName = "00";
	const int startIdx = 2670;
	const int imageCount = 120;
	const int endIdx = startIdx + imageCount;
	const string colorPostfix = ".png";
	const string depthPostfix = "_depth.png";
#endif

	const int w = 640;
	const int h = 480;

	ofstream outfile("pose.txt");
	int frameCount = 0;

	for(int imgidx=0;imgidx<=imageCount;imgidx++) {
		// process each image and perform reconstruction
		string colorImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + colorPostfix;
		string depthImageName = path + imageName + PhGUtils::toString(startIdx+imgidx) + depthPostfix;
		PhGUtils::message("Processing file " + colorImageName);
		
		vector<unsigned char> colordata = PhGUtils::fromQImage(colorImageName);
		vector<unsigned char> depthdata = PhGUtils::fromQImage(depthImageName);
		vector<float> f, pose;
		ptracker.reconstructionWithSingleFrame(&(colordata[0]), &(depthdata[0]), pose, f);
		// save the result to a file
		for(int i=0;i<pose.size();i++)
			outfile << pose[i] << ((i==pose.size()-1)?'\n':'\t');

		for(int i=0;i<pose.size();i++)
			cout << pose[i] << ((i==pose.size()-1)?'\n':'\t');


		frameCount++;
		//::system("pause");
	}
	outfile.close();

	ptracker.printStats();
}

int main(int argc, char *argv[])
{
	createConsole();
	QApplication a(argc, argv);

	glutInit(&argc, argv);

	int runType = 0;
	cout << "Please choose a run type:" << endl;
	cout << "\t" << "0. test pose tracker." << endl;
	cout << "\t" << "1. realtime tracking." << endl;
	cin >> runType;
	switch( runType ) {
	case 0:
		testPoseTracker();
		return a.exec();
	default:
		MultilinearFaceRecon w;
		w.show();
		return a.exec();
	}	
}

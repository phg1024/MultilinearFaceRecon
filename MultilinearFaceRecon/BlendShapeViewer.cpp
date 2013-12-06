#include "BlendShapeViewer.h"
#include "Utils/utility.hpp"
#include "Utils/fileutils.h"

BlendShapeViewer::BlendShapeViewer(QWidget* parent):
	GL3DCanvas(parent)
{
	this->setSceneScale(3.0);

	// load a dummy mesh
	PhGUtils::OBJLoader loader;
	loader.load("../Data/shape_0.obj");
	mesh.initWithLoader( loader );

	targetSet = false;

	loadLandmarks();

	updateMeshWithReconstructor();

	connect(&recon, SIGNAL(oneiter()), this, SLOT(updateMeshWithReconstructor()));
}


BlendShapeViewer::~BlendShapeViewer(void)
{
}

void BlendShapeViewer::initializeGL() {
	GL3DCanvas::initializeGL();
}

void BlendShapeViewer::resizeGL(int w, int h) {
	GL3DCanvas::resizeGL(w, h);
}

void BlendShapeViewer::paintGL() {
	GL3DCanvas::paintGL();

	glColor4f(0, 0.5, 1.0, 1.0);
	mesh.drawFrame();

	glColor4f(0, 0, 0, 0.25);
	targetMesh.drawFrame();

	drawLandmarks();
}

bool BlendShapeViewer::loadLandmarks()
{
	const string filename = "../Data/landmarks.txt";
	ifstream fin(filename, ios::in);
	if( fin.is_open() ) {
		landmarks.reserve(128);
		int idx;
		while(fin.good()) {
			fin >> idx;
			landmarks.push_back(idx);
		}
		PhGUtils::message("landmarks loaded.");
		cout << "total landmarks = " << landmarks.size() << endl;

		return true;
	}
	else {
		return false;
	}
}

void BlendShapeViewer::drawLandmarks() {
	glPointSize(3.0);
	glColor4f(1, 0, 0, 1);
	glBegin(GL_POINTS);
	for_each(landmarks.begin(), landmarks.end(), [&](int vidx){
		const PhGUtils::QuadMesh::vert_t& v = mesh.vertex(vidx);
		glVertex3f(v.x, v.y, v.z);
	});
	glEnd();

	if( targetSet ) {
		glPointSize(3.0);
		glColor4f(0, 1, 0, 1);
		glBegin(GL_POINTS);
		for_each(landmarks.begin(), landmarks.end(), [&](int vidx){
			const PhGUtils::QuadMesh::vert_t& v = targetMesh.vertex(vidx);
			glVertex3f(v.x, v.y, v.z);
		});
		glEnd();
	}
}

void BlendShapeViewer::updateMeshWithReconstructor() {
	cout << "updating mesh with recon ..." << endl;
	// set the vertices of mesh with the template mesh in the reconstructor
	const Tensor1<float>& tplt = recon.currentMesh();
	for(int i=0,idx=0;i<tplt.length()/3;i++) {
		mesh.vertex(i).x = tplt(idx++);
		mesh.vertex(i).y = tplt(idx++);
		mesh.vertex(i).z = tplt(idx++);
	}
	update();
}

void BlendShapeViewer::bindTargetMesh( const string& filename ) {
	PhGUtils::OBJLoader loader;
	loader.load(filename);
	targetMesh.initWithLoader( loader );

	// generate target points 
	vector<pair<PhGUtils::Point3f, int>> pts;
	for(int i=0;i<landmarks.size();i++) {
		int vidx = landmarks[i];
		const PhGUtils::Point3f& vi = targetMesh.vertex(vidx);
		pts.push_back(make_pair(vi, vidx));
	}

	recon.bindTarget(pts);
	targetSet = true;
}

void BlendShapeViewer::fit() {
	recon.fit();
	updateMeshWithReconstructor();
}

void BlendShapeViewer::generatePrior() {
	const string path = "C:\\Users\\PhG\\Desktop\\Data\\FaceWarehouse_Data_0\\";
	const string foldername = "Tester_";
	//const string meshFolder = "Blendshape";
	const string meshFolder = "TrainingPose";
	//const string meshname = "shape_";
	const string meshname = "pose_";
	const string extension = ".obj";
	const int totalTesters = 150;
	const int totalMeshes = 20;

	vector<Tensor1<float>> Wids;
	vector<Tensor1<float>> Wexps;

	for(int i=0;i<totalTesters;i++) {
		for(int j=0;j<totalMeshes;j++) {
			stringstream ss;
			ss << path << foldername << (i+1) << "\\" << meshFolder + "\\" + meshname << j << extension;

			string filename = ss.str();

			PhGUtils::message("Fitting " + filename);

			PhGUtils::OBJLoader loader;
			loader.load(filename);
			targetMesh.initWithLoader( loader );
			targetSet = true;

			// generate target points 	
			vector<pair<PhGUtils::Point3f, int>> pts;
			for(int i=0;i<landmarks.size();i++) {
				int vidx = landmarks[i];
				const PhGUtils::Point3f& vi = targetMesh.vertex(vidx);
				pts.push_back(make_pair(vi, vidx));
			}
			recon.bindTarget(pts);
			recon.fit();
			updateMeshWithReconstructor();
			QApplication::processEvents();

			Wids.push_back(recon.identityWeights());
			Wexps.push_back(recon.expressionWeights());
		}
	}

	// write the fitted weights to files
	PhGUtils::write2file(Wids, "wid.txt");
	PhGUtils::write2file(Wexps, "wexp.txt");
}


void BlendShapeViewer::keyPressEvent( QKeyEvent *e ) {
	switch( e->key() ) {
	case Qt::Key_Space:
		{
			fit();
			break;
		}
	case Qt::Key_P:
		{
			recon.togglePrior();
			break;
		}
	case Qt::Key_E:
		{
			PhGUtils::message("Please input expression prior weight:");
			float w;
			cin >> w;
			recon.expPriorWeights(w);
			break;
		}
	case Qt::Key_I: 
		{
			PhGUtils::message("Please input identity prior weight:");
			float w;
			cin >> w;
			recon.idPriorWeight(w);
			break;
		}
	}
}

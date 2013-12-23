#include "BlendShapeViewer.h"
#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"

BlendShapeViewer::BlendShapeViewer(QWidget* parent):
	GL3DCanvas(parent)
{
	this->resize(640, 480);
	this->setSceneScale(1.0);

	// load a dummy mesh
	PhGUtils::OBJLoader loader;
	loader.load("../Data/shape_0.obj");
	mesh.initWithLoader( loader );

	targetSet = false;

	loadLandmarks();

	updateMeshWithReconstructor();

	connect(&recon, SIGNAL(oneiter()), this, SLOT(updateMeshWithReconstructor()));

	mProj = PhGUtils::KinectColorProjection.transposed();
	mMv = PhGUtils::Matrix4x4f::identity();
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

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glPushMatrix();
	
	// setup the viewing matrices
	glViewport(0, 0, 640, 480);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(mProj.data());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMultMatrixf(mMv.data());
	
	enableLighting();

	glColor4f(0, 0.5, 1.0, 1.0);
	mesh.draw();
	//drawGenreatedMesh();

	glColor4f(0, 0, 0, 0.25);
	targetMesh.drawFrame();

	drawLandmarks();

	disableLighting();

	glPopMatrix();
	glDisable(GL_CULL_FACE);
}

bool BlendShapeViewer::loadLandmarks()
{
	const string filename = "../Data/model/landmarks.txt";
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

void BlendShapeViewer::drawGenreatedMesh()
{
	const Tensor1<float>& tplt = recon.currentMesh();
	// draw the mesh directly with the reconstructed data
	int nFaces = mesh.faceCount();
	glBegin(GL_QUADS);
	for(int i=0;i<nFaces;i++) {
		PhGUtils::QuadMesh::face_t& f = mesh.face(i);
		int v1idx = f.x * 3, v2idx = f.y * 3, v3idx = f.z * 3, v4idx = f.w * 3;
		PhGUtils::QuadMesh::vert_t v1(tplt(v1idx++), tplt(v1idx++), tplt(v1idx++));
		PhGUtils::QuadMesh::vert_t v2(tplt(v2idx++), tplt(v2idx++), tplt(v2idx++));
		PhGUtils::QuadMesh::vert_t v3(tplt(v3idx++), tplt(v3idx++), tplt(v3idx++));
		PhGUtils::QuadMesh::vert_t v4(tplt(v4idx++), tplt(v4idx++), tplt(v4idx++));

		// compute normal
		PhGUtils::QuadMesh::norm_t n = PhGUtils::Vector3f(v1, v2).cross(PhGUtils::Vector3f(v1, v4));
		n.normalize();

		glNormal3f(n.x, n.y, n.z);
		//glNormal3f(n1.x, n1.y, n1.z);
		glVertex3f(v1.x, v1.y, v1.z);
		//glNormal3f(n2.x, n2.y, n2.z);
		glVertex3f(v2.x, v2.y, v2.z);
		//glNormal3f(n3.x, n3.y, n3.z);
		glVertex3f(v3.x, v3.y, v3.z);
		//glNormal3f(n4.x, n4.y, n4.z);
		glVertex3f(v4.x, v4.y, v4.z);
	}
	glEnd();
}


void BlendShapeViewer::drawLandmarks() {
	glPointSize(3.0);
	glColor4f(1, 0, 0, 1);
	// landmarks on the mesh
	GLfloat mat_diffuse[] = {1.0, 0.25, 0.25, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

	glBegin(GL_POINTS);
	for_each(landmarks.begin(), landmarks.end(), [&](int vidx){
		const PhGUtils::QuadMesh::vert_t& v = mesh.vertex(vidx);
		glNormal3f(0, 0, 1.0);
		glVertex3f(v.x, v.y, v.z);
	});
	glEnd();

	if( targetSet ) {
		GLfloat mat_diffuse[] = {0.25, 1.0, 0.25, 1.0};
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

		glPointSize(3.0);

		//float minZ = numeric_limits<float>::max(), maxZ = -numeric_limits<float>::max();
		//for_each(targetLandmarks.begin(), targetLandmarks.end(), [&](const PhGUtils::Point3f p){
		//	minZ = min(p.z, minZ);
		//	maxZ = max(p.z, maxZ);
		//});
		//float diffZ = maxZ - minZ;
		//PhGUtils::debug("maxZ", maxZ, "minZ", minZ);

		glBegin(GL_POINTS);
		for_each(targetLandmarks.begin(), targetLandmarks.end(), [&](const PhGUtils::Point3f p){
			/*
			float t = (p.z -minZ) / diffZ;
			glColor4f(0, 1-t, t, 1);
			*/
			glNormal3f(0, 0, 1.0);
			glVertex3f(p.x, p.y, p.z);
		});
		glEnd();
	}
}

void BlendShapeViewer::updateMeshWithReconstructor() {
	//cout << "updating mesh with recon ..." << endl;
	// set the vertices of mesh with the template mesh in the reconstructor
	const Tensor1<float>& tplt = recon.currentMesh();

	for(int i=0,idx=0;i<tplt.length()/3;i++) {
		mesh.vertex(i).x = tplt(idx++);
		mesh.vertex(i).y = tplt(idx++);
		mesh.vertex(i).z = tplt(idx++);
	}
	update();
}

// @note	lms may have larger size than landmarks, so always use the length of landmarks
void BlendShapeViewer::bindTargetLandmarks( const vector<PhGUtils::Point3f>& lms )
{
	targetLandmarks = lms;
	vector<pair<PhGUtils::Point3f, int>> pts;
	for(int i=0;i<landmarks.size();i++) {
		int vidx = landmarks[i];
		pts.push_back(make_pair(lms[i], vidx));
	}

	recon.bindTarget(pts);
	targetSet = true;
}

void BlendShapeViewer::bindTargetMesh( const string& filename ) {
	PhGUtils::OBJLoader loader;
	loader.load(filename);
	targetMesh.initWithLoader( loader );

	// generate target points 
	vector<pair<PhGUtils::Point3f, int>> pts;
	targetLandmarks.resize(landmarks.size());
	for(int i=0;i<landmarks.size();i++) {
		int vidx = landmarks[i];
		const PhGUtils::Point3f& vi = targetMesh.vertex(vidx);
		pts.push_back(make_pair(vi, vidx));
		targetLandmarks[i] = vi;
	}

	recon.bindTarget(pts);
	targetSet = true;
}

void BlendShapeViewer::fit(MultilinearReconstructor::FittingOption ops) {
	//PhGUtils::Timer t;
	//t.tic();
	recon.fit(ops);
	//t.toc("reconstruction");
	
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
	GL3DCanvas::keyPressEvent(e);

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

void BlendShapeViewer::enableLighting()
{
	GLfloat light_position[] = {10.0,10.0,10.0,1.0};
	GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_diffuse[] = {0.75, 0.75, 0.75, 1.0};
	GLfloat mat_shininess[] = {25.0};
	GLfloat light_ambient[] = {0.05, 0.05, 0.05, 1.0};
	GLfloat white_light[] = {1.0, 1.0, 1.0, 1.0};

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
}

void BlendShapeViewer::disableLighting()
{
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);
}
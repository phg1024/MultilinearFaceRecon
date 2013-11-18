#include "BlendShapeViewer.h"
#include "Utils/utility.hpp"

BlendShapeViewer::BlendShapeViewer(QWidget* parent):
	GL3DCanvas(parent)
{
	this->setSceneScale(2.0);

	// load a dummy mesh
	OBJLoader loader;
	loader.load("../Data/shape_0.obj");

	const vector<OBJLoader::face_t>& faces = loader.getFaces();
	const vector<OBJLoader::vert_t>& verts = loader.getVerts();
	mesh.initWithLoader( loader );

	// set the vertices of mesh with the template mesh in the reconstructor
	const Tensor1<float>& tplt = recon.templateMesh();
	for(int i=0,idx=0;i<tplt.length()/3;i++) {
		mesh.vertex(i).x = tplt(idx++);
		mesh.vertex(i).y = tplt(idx++);
		mesh.vertex(i).z = tplt(idx++);
	}

	loadLandmarks();
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
		message("landmarks loaded.");
		cout << "total landmarks = " << landmarks.size() << endl;
		return true;
	}
	else {
		return false;
	}
}

void BlendShapeViewer::drawLandmarks()
{
	glPointSize(3.0);
	glColor4f(1, 0, 0, 1);
	glBegin(GL_POINTS);
	for_each(landmarks.begin(), landmarks.end(), [&](int vidx){
		const QuadMesh::vert_t& v = mesh.vertex(vidx);
		glVertex3f(v.x, v.y, v.z);
	});
	glEnd();
}

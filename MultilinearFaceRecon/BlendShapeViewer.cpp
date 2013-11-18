#include "BlendShapeViewer.h"


BlendShapeViewer::BlendShapeViewer(QWidget* parent):
	GL3DCanvas(parent)
{
	this->setSceneScale(1.5);
	OBJLoader loader;
	loader.load("../Data/shape_0.obj");

	const vector<OBJLoader::face_t>& faces = loader.getFaces();
	const vector<OBJLoader::vert_t>& verts = loader.getVerts();
	mesh.initWithLoader( loader );

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
}
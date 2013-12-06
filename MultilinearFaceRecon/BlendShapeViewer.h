#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#include "MultilinearReconstructor.h"

class BlendShapeViewer : GL3DCanvas
{
	Q_OBJECT
public:
	BlendShapeViewer(QWidget* parent);
	~BlendShapeViewer(void);

	void bindTargetMesh(const string& filename);
	void fit();
	void generatePrior();

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int w, int h);

	virtual void keyPressEvent(QKeyEvent *e);

	void drawLandmarks();

private:
	bool loadLandmarks();

private slots:
	void updateMeshWithReconstructor();	

private:
	bool targetSet;
	PhGUtils::QuadMesh mesh, targetMesh;
	vector<int> landmarks;
	MultilinearReconstructor recon;
};
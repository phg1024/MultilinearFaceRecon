#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#include "MultilinearReconstructor.h"

class BlendShapeViewer : GL3DCanvas
{
public:
	BlendShapeViewer(QWidget* parent);
	~BlendShapeViewer(void);

	void bindTargetMesh(const string& filename);
	void fit();

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int w, int h);

	void drawLandmarks();

private:
	bool loadLandmarks();
	void updateMeshWithReconstructor();

private:
	QuadMesh mesh, targetMesh;
	vector<int> landmarks;
	MultilinearReconstructor recon;
};
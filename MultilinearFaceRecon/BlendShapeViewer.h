#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"

class BlendShapeViewer : GL3DCanvas
{
public:
	BlendShapeViewer(QWidget* parent);
	~BlendShapeViewer(void);

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int w, int h);

private:
	QuadMesh mesh;
};


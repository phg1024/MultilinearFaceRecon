#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#include "MultilinearReconstructor.h"

class BlendShapeViewer : public GL3DCanvas
{
	Q_OBJECT
public:
	BlendShapeViewer(QWidget* parent);
	~BlendShapeViewer(void);

	void bindTargetMesh(const string& filename);
	void bindTargetLandmarks(const vector<PhGUtils::Point3f>& pts);
	void fit(MultilinearReconstructor::FittingOption ops = MultilinearReconstructor::FIT_ALL);
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

	vector<PhGUtils::Point3f> targetLandmarks;
	vector<int> landmarks;
	MultilinearReconstructor recon;

	PhGUtils::Matrix4x4f mProj, mMv;
};
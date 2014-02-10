#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/Mesh.h"
#include "MultilinearReconstructor.h"
#include "MultilinearReconstructorGPU.cuh"

#include <QtWidgets/QFileDialog>
#include <QGLFrameBufferObject>

class BlendShapeViewer : public GL3DCanvas
{
	Q_OBJECT
public:
	BlendShapeViewer(QWidget* parent);
	~BlendShapeViewer(void);

	void bindTargetMesh(const string& filename);
	void bindTargetLandmarks(const vector<PhGUtils::Point3f>& pts, MultilinearReconstructor::TargetType ttp = MultilinearReconstructor::TargetType_3D);
	void bindTargetLandmarksGPU( const vector<PhGUtils::Point3f>& lms, MultilinearReconstructor::TargetType ttp = MultilinearReconstructor::TargetType_2D );
	void bindRGBDTarget(
		const vector<unsigned char>& colordata,
		const vector<unsigned char>& depthdata
		);
	void bindRGBDTargetGPU(
		const vector<unsigned char>& colordata,
		const vector<unsigned char>& depthdata
		);

	enum TransferDirection{
		CPUToGPU,
		GPUToCPU
	};
	void transferParameters(TransferDirection dir);

	void fit(MultilinearReconstructor::FittingOption ops = MultilinearReconstructor::FIT_ALL);
	void fit2d(MultilinearReconstructor::FittingOption ops = MultilinearReconstructor::FIT_ALL);
	void fitICP(MultilinearReconstructor::FittingOption ops = MultilinearReconstructor::FIT_ALL);
	void fitICP_GPU(MultilinearReconstructorGPU::FittingOption ops = MultilinearReconstructorGPU::FIT_ALL);

	void generatePrior();

	const MultilinearReconstructor& getReconstructor() const {
		return recon;
	}

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int w, int h);

	void setupViewingParameters();

	void initFBO();

	// keyboard events handler
	virtual void keyPressEvent(QKeyEvent *e);

	void enableLighting();
	void disableLighting();

	void drawLandmarks();
	void drawGenreatedMesh();

	void drawMeshToFBO();

private:
	bool loadLandmarks();

private slots:
	void updateMeshWithReconstructor();	
	void updateMeshWithGPUReconstructor();	

private:
	bool targetSet;
	PhGUtils::QuadMesh mesh, targetMesh;

	vector<PhGUtils::Point3f> targetLandmarks;
	vector<int> landmarks;
	MultilinearReconstructorGPU GPURecon;
	MultilinearReconstructor recon;

	PhGUtils::Matrix4x4f mProj, mMv;

	bool showLandmarks;

private:
	// used for synthesis step
	shared_ptr<QGLFramebufferObject> fbo;
	vector<float> depthBuffer;
	vector<unsigned char> colorBuffer;
};
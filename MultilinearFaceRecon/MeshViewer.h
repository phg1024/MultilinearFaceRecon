#pragma once

#include "OpenGL/gl3dcanvas.h"
#include "Geometry/Mesh.h"
#include "Geometry/geometryutils.hpp"

#include <QImage>

class MeshViewer : public GL3DCanvas
{
public:
  MeshViewer(QWidget *parent = 0);
  ~MeshViewer();

  void bindImage(const QImage& I) {
    img = I;
    imgWidth = img.width();
    imgHeight = img.height();
    resize(imgWidth, imgHeight);
    update();
  }

  void bindMesh(PhGUtils::QuadMesh &m) {
    mesh = m;
    mesh.computeNormals();
    update();
  }

  void bindLandmarks(const vector<float> &pts) {
    landmarks = pts;
  }

  void setFocalLength(double fval) {
    fLength = fval;
  }

  void setTranslation(float x, float y, float z) {
    tx = x; ty = y; tz = z;
  }

protected:
  virtual void initializeGL();
  virtual void paintGL();
  virtual void resizeGL(int w, int h);

  void enableLighting();
  void disableLighting();

  void setupViewingParameters();
    
private:
  void drawLandmarks();
  void drawImage();

private:
  PhGUtils::Matrix4x4f mProj, mMv;

private:
  QImage img;
  PhGUtils::QuadMesh mesh;
  vector<float> landmarks;
  
  int imgWidth, imgHeight;
  double fLength;

  bool showLandmarks;
  bool showImage;

  // translation of the mesh
  float tx, ty, tz;
};


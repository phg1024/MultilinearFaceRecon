#pragma once

#include "phgutils.h"
#include "Math/Tensor.hpp"
#include "Geometry/Mesh.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/matrix.hpp"

#include <QGLWidget>
#include <QGLFramebufferObject>

class OffscreenRenderer {
public:
  OffscreenRenderer(int w, int h);

  /// off screen rendering related
  void bindFBO(const shared_ptr<QGLFramebufferObject> &f) {
    fbo = f;
  }
  void resizeBuffers(int w, int h);
  void updateMesh(const Tensor1<double> &tmesh);
  void updateFocalLength(float f);
  
  int getWidth() const { return fbo_width; }
  int getHeight() const { return fbo_height; }
  const PhGUtils::QuadMesh& getBaseMesh() const { return baseMesh; }
  PhGUtils::QuadMesh& getBaseMesh() { return baseMesh; }
  const vector<float>& getDepthMap() const { return depthMap; }
  const vector<unsigned char>& getIndexMap() const { return indexMap; }

  void render() { renderMeshToFBO(); }
  void project(double &u, double &v, double &d, double X, double Y, double Z);
protected:
  void setupViewingParametersFBO();
  void updateViewingMatricesFBO();
  void renderMeshToFBO();

  PhGUtils::QuadMesh baseMesh;
  vector<int> faceSet;

  float f;  // focal length
  PhGUtils::Matrix4x4f mProj, mMv;
  shared_ptr<QGLWidget> dummyWgt;
  shared_ptr<QGLFramebufferObject> fbo;
  int fbo_width, fbo_height;
  vector<float> depthMap;
  vector<unsigned char> indexMap;
};
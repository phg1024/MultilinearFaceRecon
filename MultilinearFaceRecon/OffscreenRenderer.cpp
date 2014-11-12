#include "OffscreenRenderer.h"

#include "Utils/utility.hpp"

OffscreenRenderer::OffscreenRenderer(int w, int h)
{
  dummyWgt = shared_ptr<QGLWidget>(new QGLWidget());
  dummyWgt->hide();
  resizeBuffers(640, 480);

  // load the default mesh
  PhGUtils::OBJLoader loader;
  loader.load("../Data/shape_0.obj");
  baseMesh.initWithLoader(loader);

  // find out interesting faces
#if 0
  faceSet.reserve(baseMesh.faceCount());
  for (int i = 0; i < baseMesh.faceCount();++i) {
    const PhGUtils::QuadMesh::face_t& f = baseMesh.face(i);
    const PhGUtils::QuadMesh::vert_t& v1 = baseMesh.vertex(f.x);
    const PhGUtils::QuadMesh::vert_t& v2 = baseMesh.vertex(f.y);
    const PhGUtils::QuadMesh::vert_t& v3 = baseMesh.vertex(f.z);
    const PhGUtils::QuadMesh::vert_t& v4 = baseMesh.vertex(f.w);
    PhGUtils::QuadMesh::vert_t center = 0.25 * (v1 + v2 + v3 + v4);
    if (center.z >= -0.3) {
      faceSet.push_back(i);
    }
  }
#else
  faceSet.reserve(baseMesh.faceCount());
  ifstream fin("../Data/facemask.txt");
  while (fin) {
    int idx;
    fin >> idx;
    faceSet.push_back(idx);
  }
  fin.close();
  cout << "faces in the mask = " << faceSet.size() << endl;
#endif
}

void OffscreenRenderer::renderMeshToFBO()
{
  dummyWgt->makeCurrent();
  fbo->bind();

#if FBO_DEBUG
  cout << (fbo->isBound() ? "bounded." : "not bounded.") << endl;
  cout << (fbo->isValid() ? "valid." : "invalid.") << endl;
#endif

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glPushMatrix();

  // setup viewing parameters
  setupViewingParametersFBO();

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glShadeModel(GL_SMOOTH);

  baseMesh.drawFaceIndices(faceSet);

  glReadPixels(0, 0, fbo_width, fbo_height, GL_DEPTH_COMPONENT, GL_FLOAT, &(depthMap[0]));
#if FBO_DEBUG
  GLenum errcode = glGetError();
  if (errcode != GL_NO_ERROR) {
    const GLubyte *errString = gluErrorString(errcode);
    fprintf(stderr, "OpenGL Error: %s\n", errString);
  }
#endif

  glReadPixels(0, 0, fbo_width, fbo_height, GL_RGBA, GL_UNSIGNED_BYTE, &(indexMap[0]));
#if FBO_DEBUG
  errcode = glGetError();
  if (errcode != GL_NO_ERROR) {
    const GLubyte *errString = gluErrorString(errcode);
    fprintf(stderr, "OpenGL Error: %s\n", errString);
  }
#endif

  glPopMatrix();

  glDisable(GL_CULL_FACE);

  fbo->release();
  dummyWgt->doneCurrent();

#if FBO_DEBUG
  ofstream fout("fbodepth.txt");
  PhGUtils::print2DArray(&(depthMap[0]), fbo_height, fbo_width, fout);
  fout.close();

  QImage img = PhGUtils::toQImage(&(indexMap[0]), fbo_width, fbo_height);
  img.save("fbo.png");
#endif
}

void OffscreenRenderer::updateMesh(const Tensor1<float> &tmesh)
{
  for (int i = 0; i < tmesh.length() / 3; i++) {
    int idx = i * 3;
    baseMesh.vertex(i).x = tmesh(idx++);
    baseMesh.vertex(i).y = tmesh(idx++);
    baseMesh.vertex(i).z = tmesh(idx);
  }
}

void OffscreenRenderer::resizeBuffers(int w, int h)
{
  fbo_width = w;
  fbo_height = h;

  dummyWgt->makeCurrent();
  fbo = shared_ptr<QGLFramebufferObject>(new QGLFramebufferObject(fbo_width, fbo_height, QGLFramebufferObject::Depth));
  dummyWgt->doneCurrent();

  depthMap.resize(fbo_width * fbo_height);
  indexMap.resize(fbo_width * fbo_height * 4);
}

void OffscreenRenderer::updateFocalLength(float focal_length)
{
  f = focal_length;
}

void OffscreenRenderer::setupViewingParametersFBO()
{
  updateViewingMatricesFBO();

  glViewport(0, 0, fbo_width, fbo_height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMultMatrixf(mProj.data());

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMultMatrixf(mMv.data());
}

void OffscreenRenderer::updateViewingMatricesFBO()
{
  double fx = fbo_width / 2.0;
  double fy = fbo_height / 2.0;
  mProj = PhGUtils::Matrix4x4f(f / fx, 0, 0, 0,
    0, f / fy, 0, 0,
    0, 0, -100.0001 / 99.9999, -0.02 / 99.9999,
    0, 0, -1.0, 0).transposed();
  mMv = PhGUtils::Matrix4x4f::identity();

}

void OffscreenRenderer::project(double &u, double &v, double &d, double X, double Y, double Z)
{
  double modelMatrix[16], projMatrix[16];
  int viewport[4];
  dummyWgt->makeCurrent();
  glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
  glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
  glGetIntegerv(GL_VIEWPORT, viewport);
  gluProject(X, Y, Z, modelMatrix, projMatrix, viewport, &u, &v, &d);
  dummyWgt->doneCurrent();
}

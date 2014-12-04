#include "MeshViewer.h"


MeshViewer::MeshViewer(QWidget *parent):GL3DCanvas(parent) {
  imgWidth = 640; imgHeight = 480;
  this->resize(imgWidth, imgHeight);
  this->setSceneScale(1.0);
  showLandmarks = false;

  fLength = 1000.0;
  mMv = PhGUtils::Matrix4x4f::identity();
}

MeshViewer::~MeshViewer() {
}

void MeshViewer::initializeGL() {
  makeCurrent();
  GL3DCanvas::initializeGL();
  doneCurrent();
}

void MeshViewer::paintGL() {
  glClearColor(1, 1, 1, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (showImage) drawImage();

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glPushMatrix();
  // setup the viewing matrices
  setupViewingParameters();

  enableLighting();

  glPushMatrix();

  float* rotationMatrix = this->trackBall.getMatrix();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(tx, ty, tz);
  glMultMatrixf(rotationMatrix);
  glTranslatef(-tx, -ty, -tz);

  glColor4f(0, 0.5, 1.0, 1.0);
  mesh.draw();

  glPopMatrix();

  if (showLandmarks) drawLandmarks();
  disableLighting();

  glPopMatrix();
  glDisable(GL_CULL_FACE);

  glDisable(GL_BLEND);
}

void MeshViewer::resizeGL(int w, int h) {
  GL3DCanvas::resizeGL(w, h);
}

void MeshViewer::setupViewingParameters() {
  glViewport(0, 0, imgWidth, imgHeight);

  /// obtain the projection matrix from recon
  double f = fLength;
  double fx = imgWidth / 2.0;
  double fy = imgHeight / 2.0;
  mProj = PhGUtils::Matrix4x4f(f / fx, 0, 0, 0,
    0, f / fy, 0, 0,
    0, 0, -100.0001 / 99.9999, -0.02 / 99.9999,
    0, 0, -1.0, 0).transposed();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glLoadMatrixf(mProj.data());

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glLoadMatrixf(mMv.data());
}

void MeshViewer::enableLighting() {
  GLfloat light_position[] = { 10.0, 0.0, 10.0, 1.0 };
  GLfloat mat_specular[] = { 0.025, 0.025, 0.025, 1.0 };
  GLfloat mat_diffuse[] = { 0.375, 0.375, 0.375, 1.0 };
  GLfloat mat_shininess[] = { 25.0 };
  GLfloat light_ambient[] = { 0.05, 0.05, 0.05, 1.0 };
  GLfloat white_light[] = { 1.0, 1.0, 1.0, 1.0 };

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

  light_position[0] = -10.0;
  glLightfv(GL_LIGHT1, GL_POSITION, light_position);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT1, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
}

void MeshViewer::disableLighting() {
  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHTING);
}

void MeshViewer::drawImage() {

}

void MeshViewer::drawLandmarks() {

}




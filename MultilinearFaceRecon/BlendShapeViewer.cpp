#include "BlendShapeViewer.h"
#include "Utils/utility.hpp"
#include "Utils/fileutils.h"
#include "Utils/Timer.h"
#include "Kinect/KinectUtils.h"

BlendShapeViewer::BlendShapeViewer(QWidget* parent) :
GL3DCanvas(parent)
{
  imageWidth = 640; imageHeight = 480;
  this->resize(imageWidth, imageHeight);
  this->setSceneScale(1.0);

  showLandmarks = false;

  // load a dummy mesh
  PhGUtils::OBJLoader loader;
  loader.load("../Data/shape_0.obj");
  mesh.initWithLoader(loader);

  //recon.setBaseMesh(mesh);

#if USE_GPU_RECON
  GPURecon.setBaseMesh(mesh);
#endif

  recon_2d.params.w_prior_exp = 0.5;

  targetSet = false;

  loadLandmarks();

  updateMeshWithReconstructor();

  //connect(&recon, SIGNAL(oneiter()), this, SLOT(updateMeshWithReconstructor()));

  mProj = PhGUtils::KinectColorProjection.transposed();
  mMv = PhGUtils::Matrix4x4f::identity();
}


BlendShapeViewer::~BlendShapeViewer(void)
{
}

void BlendShapeViewer::initializeGL() {
  GL3DCanvas::initializeGL();
  initFBO();
  //recon.bindFBO(fbo);
}

void BlendShapeViewer::resizeGL(int w, int h) {
  GL3DCanvas::resizeGL(w, h);
}

void BlendShapeViewer::paintGL() {
  makeCurrent();

  glClearColor(1, 1, 1, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glPushMatrix();

  // setup the viewing matrices
  setupViewingParameters();

  enableLighting();

  glPushMatrix();

  float tx = recon_2d.params.RTparams[3], ty = recon_2d.params.RTparams[4], tz = recon_2d.params.RTparams[5];
  float* rotationMatrix = this->trackBall.getMatrix();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(tx, ty, tz);
  glMultMatrixf(rotationMatrix);
  glTranslatef(-tx, -ty, -tz);

  glColor4f(0, 0.5, 1.0, 1.0);
  mesh.draw();

  glPopMatrix();

  //mesh.drawFrame();
  //drawGenreatedMesh();

  glColor4f(0, 0, 0, 0.25);
  targetMesh.drawFrame();

  if (showLandmarks)
    drawLandmarks();

  disableLighting();

  //drawMeshToFBO();

  glPopMatrix();
  glDisable(GL_CULL_FACE);
}

void BlendShapeViewer::setupViewingParameters() {
  glViewport(0, 0, imageWidth, imageHeight);

  /// obtain the projection matrix from recon
  //double f = recon.cameraFocalLength();
  double f = recon_2d.params.camparams.fy;
  double fx = imageWidth / 2.0;
  double fy = imageHeight / 2.0;
  mProj = PhGUtils::Matrix4x4f(f / fx, 0, 0, 0,
    0, f / fy, 0, 0,
    0, 0, -1000.00001 / 999.99999, -0.02 / 999.99999,
    0, 0, -1.0, 0).transposed();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMultMatrixf(mProj.data());

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMultMatrixf(mMv.data());
}

bool BlendShapeViewer::loadLandmarks()
{
  const string filename = "../Data/model/landmarks.txt";
  ifstream fin(filename, ios::in);
  if (fin.is_open()) {
    landmarks.reserve(128);
    int idx;
    while (fin.good()) {
      fin >> idx;
      landmarks.push_back(idx);
    }
    PhGUtils::message("landmarks loaded.");
    cout << "total landmarks = " << landmarks.size() << endl;

    return true;
  }
  else {
    return false;
  }
}

void BlendShapeViewer::drawGenreatedMesh()
{
#if 0
  const Tensor1<float>& tplt = recon.currentMesh();
  // draw the mesh directly with the reconstructed data
  int nFaces = mesh.faceCount();
  glBegin(GL_QUADS);
  for(int i=0;i<nFaces;i++) {
    PhGUtils::QuadMesh::face_t& f = mesh.face(i);
    int v1idx = f.x * 3, v2idx = f.y * 3, v3idx = f.z * 3, v4idx = f.w * 3;
    PhGUtils::QuadMesh::vert_t v1(tplt(v1idx++), tplt(v1idx++), tplt(v1idx++));
    PhGUtils::QuadMesh::vert_t v2(tplt(v2idx++), tplt(v2idx++), tplt(v2idx++));
    PhGUtils::QuadMesh::vert_t v3(tplt(v3idx++), tplt(v3idx++), tplt(v3idx++));
    PhGUtils::QuadMesh::vert_t v4(tplt(v4idx++), tplt(v4idx++), tplt(v4idx++));

    // compute normal
    PhGUtils::QuadMesh::norm_t n = PhGUtils::Vector3f(v1, v2).cross(PhGUtils::Vector3f(v1, v4));
    n.normalize();

    glNormal3f(n.x, n.y, n.z);
    //glNormal3f(n1.x, n1.y, n1.z);
    glVertex3f(v1.x, v1.y, v1.z);
    //glNormal3f(n2.x, n2.y, n2.z);
    glVertex3f(v2.x, v2.y, v2.z);
    //glNormal3f(n3.x, n3.y, n3.z);
    glVertex3f(v3.x, v3.y, v3.z);
    //glNormal3f(n4.x, n4.y, n4.z);
    glVertex3f(v4.x, v4.y, v4.z);
  }
  glEnd();
#endif
}


void BlendShapeViewer::drawLandmarks() {
  glPointSize(5.0);
  glColor4f(1, 0, 0, 1);
  // landmarks on the mesh
  GLfloat mat_diffuse[] = { 1.0, 0.25, 0.25, 1.0 };
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  double modelMatrix[16], projMatrix[16];
  int viewport[4];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
  glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
  glGetIntegerv(GL_VIEWPORT, viewport);

  glBegin(GL_POINTS);
  int qidx = 0;
  for_each(landmarks.begin(), landmarks.end(), [&](int vidx){
    const PhGUtils::QuadMesh::vert_t& v = mesh.vertex(vidx);
    glNormal3f(0, 0, 1.0);
    //glVertex3f(v.x, v.y, v.z);

    double x, y, z;
    gluProject(v.x, v.y, v.z, modelMatrix, projMatrix, viewport, &x, &y, &z);

    double nx, ny, nz;
    gluUnProject(x, y, 0.999975, modelMatrix, projMatrix, viewport, &nx, &ny, &nz);

    glNormal3f(0, 0, 1.0);
    glVertex3f(nx, ny, nz);

    /*
    if (targetSet) {
    PhGUtils::Point3f q2d;
    PhGUtils::worldToColor(v.x, v.y, v.z, q2d.x, q2d.y, q2d.z,
    recon_2d.params.camparams.fx, recon_2d.params.camparams.cx, recon_2d.params.camparams.cy);

    cout << q2d << " | " << targetLandmarks[qidx] << endl;
    ++qidx;
    }
    */
  });
  glEnd();

  if (targetSet) {
    GLfloat mat_diffuse[] = { 0.25, 1.0, 0.25, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

    glPointSize(3.0);

    //float minZ = numeric_limits<float>::max(), maxZ = -numeric_limits<float>::max();
    //for_each(targetLandmarks.begin(), targetLandmarks.end(), [&](const PhGUtils::Point3f p){
    //	minZ = min(p.z, minZ);
    //	maxZ = max(p.z, maxZ);
    //});
    //float diffZ = maxZ - minZ;
    //PhGUtils::debug("maxZ", maxZ, "minZ", minZ);

    glBegin(GL_POINTS);
    for_each(targetLandmarks.begin(), targetLandmarks.end(), [&](const PhGUtils::Point3f p){
      /*
      float t = (p.z -minZ) / diffZ;
      glColor4f(0, 1-t, t, 1);
      */
      /*
      PhGUtils::Point3f q;
      PhGUtils::colorToWorld(p.x, p.y, 1.0, q.x, q.y, q.z,
      recon_2d.params.camparams.fx, recon_2d.params.camparams.cx, recon_2d.params.camparams.cy);
      */
      double x, y, z;
      gluUnProject(p.x, height() - p.y, 0.99995, modelMatrix, projMatrix, viewport, &x, &y, &z);
      glNormal3f(0, 0, 1.0);
      glVertex3f(x, y, z);
    });
    glEnd();
  }
}

#if USE_GPU_RECON
void BlendShapeViewer::updateMeshWithGPUReconstructor() {
  //cout << "updating mesh with recon ..." << endl;
  // set the vertices of mesh with the template mesh in the reconstructor
  const Tensor1<float>& tplt = GPURecon.currentMesh();

  //cout << tplt.length() << endl;
  for(int i=0,idx=0;i<tplt.length()/3;i++) {
    mesh.vertex(i).x = tplt(idx++);
    mesh.vertex(i).y = tplt(idx++);
    mesh.vertex(i).z = tplt(idx++);
  }
  update();
  //::system("pause");
}
#endif

void BlendShapeViewer::updateMeshWithReconstructor() {
  //cout << "updating mesh with recon ..." << endl;
  // set the vertices of mesh with the template mesh in the reconstructor
#if 0
  const Tensor1<float>& tplt = recon.currentMesh();
#else
  const Tensor1<float>& tplt = recon_2d.currentMesh();
#endif

  for (int i = 0, idx = 0; i < tplt.length() / 3; i++) {
    mesh.vertex(i).x = tplt(idx++);
    mesh.vertex(i).y = tplt(idx++);
    mesh.vertex(i).z = tplt(idx++);
  }

  mesh.computeNormals();

  /*
  PhGUtils::OBJWriter writer;
  writer.save(mesh, "transformed_mesh.obj");
  */

  update();
}

#if USE_GPU_RECON
void BlendShapeViewer::bindTargetLandmarksGPU( const vector<PhGUtils::Point3f>& lms, MultilinearReconstructor_old::TargetType ttp ) {
  switch( ttp ) {
  case MultilinearReconstructor_old::TargetType_2D:
  {
    targetLandmarks.clear();
    for(int i=0;i<lms.size();i++) {
      PhGUtils::Point3f p;
      PhGUtils::colorToWorld(lms[i].x, lms[i].y, lms[i].z, p.x, p.y, p.z);
      targetLandmarks.push_back(p);
    }
    break;
  }
  case MultilinearReconstructor_old::TargetType_3D:
  {
    targetLandmarks = lms;
    break;
  }
  }

  GPURecon.bindTarget(lms);
  targetSet = true;
}
#endif

// @note	lms may have larger size than landmarks, so always use the length of landmarks
void BlendShapeViewer::bindTargetLandmarks(const vector<PhGUtils::Point3f>& lms, MultilinearReconstructor_old::TargetType ttp)
{
  switch (ttp) {
  case MultilinearReconstructor_old::TargetType_2D:
  {
    targetLandmarks.clear();
    for (int i = 0; i < lms.size(); i++) {
#if 0
      PhGUtils::Point3f p;
      PhGUtils::colorToWorld(lms[i].x, lms[i].y, lms[i].z, p.x, p.y, p.z);
      targetLandmarks.push_back(p);
#else
      targetLandmarks.push_back(lms[i]);
#endif
    }
    break;
  }
  case MultilinearReconstructor_old::TargetType_3D:
  {
    targetLandmarks = lms;
    break;
  }
  }

  // bind the target locations with vertex indices
  vector<pair<PhGUtils::Point3f, int>> pts;
  for (int i = 0; i < landmarks.size(); i++) {
    int vidx = landmarks[i];
    pts.push_back(make_pair(lms[i], vidx));
  }

  // update the constraints
  constraints.resize(landmarks.size());
  for (int i = 0; i < landmarks.size(); ++i) {
    constraints[i].vidx = landmarks[i];
    constraints[i].weight = 1.0;
    constraints[i].q.x = lms[i].x;
    constraints[i].q.y = lms[i].y;
  }

  // pass the targets to recon
  // recon.bindTarget(pts, ttp);
  targetSet = true;
}

void BlendShapeViewer::bindRGBDTarget(const vector<unsigned char>& colordata, const vector<unsigned char>& depthdata)
{
  //recon.bindRGBDTarget(colordata, depthdata);
}

#if USE_GPU_RECON
void BlendShapeViewer::bindRGBDTargetGPU(const vector<unsigned char>& colordata, const vector<unsigned char>& depthdata)
{
  GPURecon.bindRGBDTarget(colordata, depthdata);
}
#endif

void BlendShapeViewer::bindTargetMesh(const string& filename) {
  PhGUtils::OBJLoader loader;
  loader.load(filename);
  targetMesh.initWithLoader(loader);

  // generate target points 
  vector<pair<PhGUtils::Point3f, int>> pts;
  targetLandmarks.resize(landmarks.size());
  for (int i = 0; i < landmarks.size(); i++) {
    int vidx = landmarks[i];
    const PhGUtils::Point3f& vi = targetMesh.vertex(vidx);
    pts.push_back(make_pair(vi, vidx));
    targetLandmarks[i] = vi;
  }

  //recon.bindTarget(pts);
  targetSet = true;
}

void BlendShapeViewer::fit(MultilinearReconstructor_old::FittingOption ops) {
  //PhGUtils::Timer t;
  //t.tic();
  //recon.fit(ops);
  //t.toc("reconstruction");

  updateMeshWithReconstructor();
}

void BlendShapeViewer::fit2d(MultilinearReconstructor_old::FittingOption ops /*= MultilinearReconstructor::FIT_ALL*/)
{
  //PhGUtils::Timer t;
  //t.tic();
#if 0
  recon.fit2d(ops);
#else
  //cout << "fitting using the new engine..." << endl;
  recon_2d.fit(constraints, FittingOption(ops));
#endif
  //t.toc("reconstruction");

  updateMeshWithReconstructor();
}


void BlendShapeViewer::fitICP(MultilinearReconstructor_old::FittingOption ops /*= MultilinearReconstructor::FIT_ALL*/)
{
  PhGUtils::Timer t;
  t.tic();
  //recon.fitICP(ops);
  t.toc("reconstruction");

  updateMeshWithReconstructor();
}

#if USE_GPU_RECON
void BlendShapeViewer::printStatsGPU() {
  GPURecon.printStats();
}
#endif

#if USE_GPU_RECON
void BlendShapeViewer::fitICP_GPU(MultilinearReconstructorGPU::FittingOption ops /*= MultilinearReconstructorGPU::FIT_ALL*/)
{
  PhGUtils::Timer t;
  t.tic();
  GPURecon.fit(ops);
  t.toc("reconstruction");

  updateMeshWithGPUReconstructor();
}
#endif

void BlendShapeViewer::generatePrior() {
#if 0
  const string path = "C:\\Users\\Peihong\\Desktop\\Data\\FaceWarehouse_Data_0\\";
  const string foldername = "Tester_";
  //const string meshFolder = "Blendshape";
  const string meshFolder = "TrainingPose";
  //const string meshname = "shape_";
  const string meshname = "pose_";
  const string extension = ".obj";
  const int totalTesters = 150;
  const int totalMeshes = 20;

  vector<Tensor1<float>> Wids;
  vector<Tensor1<float>> Wexps;

  for(int i=0;i<totalTesters;i++) {
    for(int j=0;j<totalMeshes;j++) {
      stringstream ss;
      ss << path << foldername << (i+1) << "\\" << meshFolder + "\\" + meshname << j << extension;

      string filename = ss.str();

      PhGUtils::message("Fitting " + filename);

      PhGUtils::OBJLoader loader;
      loader.load(filename);
      targetMesh.initWithLoader( loader );
      targetSet = true;

      // generate target points 	
      vector<pair<PhGUtils::Point3f, int>> pts;
      for(int i=0;i<landmarks.size();i++) {
        int vidx = landmarks[i];
        const PhGUtils::Point3f& vi = targetMesh.vertex(vidx);
        pts.push_back(make_pair(vi, vidx));
      }
      recon.bindTarget(pts);
      recon.fit();
      updateMeshWithReconstructor();
      QApplication::processEvents();

      Wids.push_back(recon.identityWeights());
      Wexps.push_back(recon.expressionWeights());
    }
  }

  // write the fitted weights to files
  PhGUtils::write2file(Wids, "wid.txt");
  PhGUtils::write2file(Wexps, "wexp.txt");
#endif
}

void BlendShapeViewer::keyPressEvent(QKeyEvent *e) {
  GL3DCanvas::keyPressEvent(e);

  switch (e->key()) {
  case Qt::Key_Escape:
  {
    cudaDeviceReset();
    exit(0);
  }
  case Qt::Key_Space:
  {
    fit();
    break;
  }
  case Qt::Key_E:
  {
    PhGUtils::message("Please input expression prior weight:");
    float w;
    cin >> w;
    //recon.expPriorWeights(w);
    break;
  }
#if USE_GPU_RECON
  case Qt::Key_G:
  {
    PhGUtils::message("Triggered GPU recon!");
    GPURecon.fit(MultilinearReconstructorGPU::FIT_POSE);
    break;
  }
#endif
  case Qt::Key_I:
  {
    PhGUtils::message("Please input identity prior weight:");
    float w;
    cin >> w;
    //recon.idPriorWeight(w);
    break;
  }
  case Qt::Key_L:
  {
    showLandmarks = !showLandmarks;
    update();
    break;
  }
  case Qt::Key_M:
  {
    vector<pair<int, int>> hints;	// index on the target mesh - index on the template mesh
    hints.push_back(make_pair(630, 9539));
    hints.push_back(make_pair(28080, 290));
    hints.push_back(make_pair(24738, 663));
    hints.push_back(make_pair(20075, 2095));
    hints.push_back(make_pair(20704, 200));
    hints.push_back(make_pair(8247, 1613));
    hints.push_back(make_pair(4358, 399));
    hints.push_back(make_pair(19395, 1889));
    hints.push_back(make_pair(181, 539));
    hints.push_back(make_pair(20981, 2179));

    hints.push_back(make_pair(20712, 544));
    hints.push_back(make_pair(5785, 2176));
    hints.push_back(make_pair(15314, 1601));
    hints.push_back(make_pair(20574, 1592));
    hints.push_back(make_pair(9333, 698));
    hints.push_back(make_pair(15240, 2123));

    QString filename = QFileDialog::getOpenFileName(this, "Please select a mesh file", "../Data/", "*.obj");

    //recon.fitMesh(filename.toStdString(), hints);
  }
  case Qt::Key_P:
  {
    //recon.togglePrior();
    break;
  }
  case Qt::Key_R:
  {
    //switchInteractionState();
    trackBall.reset();
    update();
    break;
  }
  }
}

void BlendShapeViewer::enableLighting()
{
  GLfloat light_position[] = { 10.0, 4.0, 10.0, 1.0 };
  GLfloat mat_specular[] = { 0.25, 0.25, 0.25, 1.0 };
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

void BlendShapeViewer::disableLighting()
{
  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHTING);
}

void BlendShapeViewer::initFBO()
{
  // attach a depth buffer
  fbo = shared_ptr<QGLFramebufferObject>(new QGLFramebufferObject(640, 480, QGLFramebufferObject::Depth));
  depthBuffer.resize(640 * 480);
  colorBuffer.resize(640 * 480 * 4);
}

void BlendShapeViewer::drawMeshToFBO()
{
  fbo->bind();

  cout << (fbo->isBound() ? "bounded." : "not bounded.") << endl;
  cout << (fbo->isValid() ? "valid." : "invalid.") << endl;

  glPushMatrix();

  setupViewingParameters();

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glDepthMask(GL_TRUE);

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glShadeModel(GL_SMOOTH);

  mesh.drawFaceIndices();


  glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, &(depthBuffer[0]));
  GLenum errcode = glGetError();
  if (errcode != GL_NO_ERROR) {
    const GLubyte *errString = gluErrorString(errcode);
    fprintf(stderr, "OpenGL Error: %s\n", errString);
  }

  glReadPixels(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, &(colorBuffer[0]));
  errcode = glGetError();
  if (errcode != GL_NO_ERROR) {
    const GLubyte *errString = gluErrorString(errcode);
    fprintf(stderr, "OpenGL Error: %s\n", errString);
  }

  glPopMatrix();

  fbo->release();


  ofstream fout("fbodepth.txt");
  PhGUtils::print2DArray(&(depthBuffer[0]), 480, 640, fout);
  fout.close();

  QImage img = PhGUtils::toQImage(&(colorBuffer[0]), 640, 480);
  img.save("fbo.png");
}

#if USE_GPU_RECON
void BlendShapeViewer::transferParameters(TransferDirection dir)
{
  switch(dir) {
  case CPUToGPU:
  {
    // rigid transformation parameters
    GPURecon.setPose( recon.getPose() );
    // identity weights
    GPURecon.setIdentityWeights( recon.identityWeights() );
    // expression weights
    GPURecon.setExpressionWeights( recon.expressionWeights() );
    break;
  }
  case GPUToCPU:
  {
    // rigid transformation parameters
    // identity weights
    // expression weights
    break;
  }
  default:
    break;
  }
}
#endif

void BlendShapeViewer::printStats()
{
  //recon.printStats();
}

void BlendShapeViewer::setReconstructionImageSize(int w, int h)
{
  imageWidth = w;
  imageHeight = h;
  recon_2d.params.camparams.cx = imageWidth / 2.0;
  recon_2d.params.camparams.cy = imageHeight / 2.0;
  recon_2d.params.camparams.fx = -500.0;
  recon_2d.params.camparams.fy = 500.0;
}

void BlendShapeViewer::resetReconstructor()
{
  // reset pose and identity
  recon_2d.reset();
}
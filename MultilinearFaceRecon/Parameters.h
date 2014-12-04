#pragma once

#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include "Math/denseblas.h"
#include "Utils/utility.hpp"

#include "MultilinearModel.h"

struct CameraParams {
  CameraParams() :fx(-5.25), fy(5.25), cx(320.0), cy(240.0){}
  double fx, fy, cx, cy;  /// estimated projection matrix
};

struct MultilinearModelParameters {
  // parameters to optimize
  static const int nRTparams = 6;

  Tensor1<double> Wid, Wexp;
  double RTparams[nRTparams];
  arma::mat R;
  arma::vec T;
  PhGUtils::Matrix3x3d Rmat;
  PhGUtils::Point3d Tvec;

  CameraParams camparams;
};

struct ReconstructionParameters {
  int imgWidth, imgHeight;

  Tensor1<double> tmesh;
  PhGUtils::QuadMesh baseMesh;
  vector<Constraint_2D> cons;

  MultilinearModel<double> model_projected;
};

struct MultilinearModelPriorParameters {
  // weights prior
  // average identity and neutral expression
  arma::vec mu_wid0, mu_wexp0;
  // mean wid and mean wexp
  arma::vec mu_wid_orig, mu_wexp_orig;
  // mean wid and mean wexp, multiplied by the inverse of sigma_?
  arma::vec mu_wid, mu_wexp;
  // actually inv(sigma_wid), inv(sigma_wexp), same for below
  arma::mat sigma_wid, sigma_wexp;

  double w_prior_id, w_prior_exp;

  void load(const string &filename_id, const string &filename_exp) {

    // the prior data is stored in the following format
    // the matrices are stored in column major order

    // ndims
    // mean vector
    // covariance matrix

    cout << "loading prior data ..." << endl;
    const string fnwid = filename_id;
    ifstream fwid(fnwid, ios::in | ios::binary);

    int ndims;
    fwid.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "identity prior dim = " << ndims << endl;

    mu_wid0.resize(ndims);
    mu_wid.resize(ndims);
    sigma_wid.resize(ndims, ndims);

    fwid.read(reinterpret_cast<char*>(mu_wid0.memptr()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(mu_wid.memptr()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(sigma_wid.memptr()), sizeof(double)*ndims*ndims);

    fwid.close();

    PhGUtils::message("identity prior loaded.");
    PhGUtils::message("processing identity prior.");
    //mu_wid.print("mean_wid");
    //sigma_wid.print("sigma_wid");
    sigma_wid = arma::inv(sigma_wid);
    mu_wid_orig = mu_wid;
    mu_wid = sigma_wid * mu_wid;
    PhGUtils::message("done");

    const string fnwexp = filename_exp;
    ifstream fwexp(fnwexp, ios::in | ios::binary);

    fwexp.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "expression prior dim = " << ndims << endl;

    mu_wexp0.resize(ndims);
    mu_wexp.resize(ndims);
    sigma_wexp.resize(ndims, ndims);

    fwexp.read(reinterpret_cast<char*>(mu_wexp0.memptr()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(mu_wexp.memptr()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(sigma_wexp.memptr()), sizeof(double)*ndims*ndims);

    fwexp.close();
    //mu_wexp.print("mean_wexp");
    //sigma_wexp.print("sigma_wexp");
    sigma_wexp = arma::inv(sigma_wexp);
    mu_wexp_orig = mu_wexp;
    mu_wexp = sigma_wexp * mu_wexp;
  }
};

struct ConvergenceParameters {
  // not used yet
};

struct SingleImageParameters {
  SingleImageParameters() {
    priorParams.w_prior_id = 0.25;
    priorParams.w_prior_exp = 0.25;
  }
  
  bool fitExpression;

  // priors
  MultilinearModelPriorParameters priorParams;

  // parameters to optimize
  MultilinearModelParameters modelParams;  

  // reconstruction related parameters
  ReconstructionParameters reconParams;

  // convergence criteria
  ConvergenceParameters convParams;

  void loadPrior(const string &filename_id, const string &filename_exp) {
    priorParams.load(filename_id, filename_exp);
  }
  void init(int n) {
    PhGUtils::message("initializing weights ...");

    // dimension of the identity weights
    int ndims_id = priorParams.mu_wid0.n_elem;
    // dimension of the expression weights
    int ndims_exp = priorParams.mu_wexp0.n_elem;

    modelParams.Wid.resize(ndims_id);
    modelParams.Wexp.resize(ndims_exp);

    // use the first person initially
    for (int i = 0; i < ndims_id; i++) {
      modelParams.Wid(i) = priorParams.mu_wid0(i);
    }

    // use neutral face initially
    for (int i = 0; i < ndims_exp; i++) {
      modelParams.Wexp(i) = priorParams.mu_wexp0(i);
    }

    PhGUtils::message("done.");
  }
  void setBaseMesh(const string &filename) {
    PhGUtils::OBJLoader loader;
    loader.load(filename);
    reconParams.baseMesh.initWithLoader(loader);
  }
  void updateMesh(MultilinearModel<double> &m) {
    m.applyWeights(modelParams.Wid, modelParams.Wexp);
    reconParams.tmesh = m.tm;
  }
  void generateFittedMesh(MultilinearModel<double> &m, bool applyWeights = true) {
    if (applyWeights) {
      //modelParams.Wid.print("Wid");
      m.updateTM0(modelParams.Wid);
      m.updateTM1(modelParams.Wexp);
      m.updateTMWithMode0(modelParams.Wexp);
    }

    auto &tplt = m.tm;
    int nverts = tplt.length() / 3;
    arma::mat pt(3, nverts);
    for (int i = 0, idx = 0; i < nverts; i++, idx += 3) {
      pt(0, i) = tplt(idx);
      pt(1, i) = tplt(idx + 1);
      pt(2, i) = tplt(idx + 2);
    }

    auto &tmesh = reconParams.tmesh;
    // batch rotation processing
    arma::mat pt_trans = modelParams.R * pt;
#pragma omp parallel for
    for (int i = 0; i < nverts; i++) {
      int idx = i * 3;
      tmesh(idx) = pt_trans(0, i) + modelParams.T(0);
      tmesh(idx + 1) = pt_trans(1, i) + modelParams.T(1);
      tmesh(idx + 2) = pt_trans(2, i) + modelParams.T(2);
    }

    tmesh.write("0.mesh");

    auto &baseMesh = reconParams.baseMesh;
    for (int i = 0; i < tmesh.length() / 3; i++) {
      int idx = i * 3;
      baseMesh.vertex(i).x = tmesh(idx++);
      baseMesh.vertex(i).y = tmesh(idx++);
      baseMesh.vertex(i).z = tmesh(idx);
    }
  }
  void updateContourPoints(const vector<vector<int>> &contourPoints,
    const vector<int> &contourVerts) {
    auto &baseMesh = reconParams.baseMesh;
    baseMesh.computeNormals(contourVerts);

    PhGUtils::Vector3f viewVec(0, 0, -1);

    // for each row, pick the best candidates
    int nContourRows = contourPoints.size();
    vector<int> candidates;
    for (int i = 0; i < nContourRows; ++i) {
      int minIdx = 0;
      float minProd = fabs(baseMesh.normal(contourPoints[i].front()).dot(viewVec));
      for (int j = 1; j < contourPoints[i].size(); ++j) {
        auto &nij = baseMesh.normal(contourPoints[i][j]);
        auto &vij = baseMesh.vertex(contourPoints[i][j]);
        viewVec = PhGUtils::Vector3f(
          vij.x,
          vij.y,
          vij.z
          ).normalized();
        float prod = fabs(nij.dot(viewVec));
        if (prod < minProd) {
          minIdx = j;
          minProd = prod;
        }
      }

      candidates.push_back(contourPoints[i][minIdx]);
      if (minIdx < contourPoints[i].size() - 1)
        candidates.push_back(contourPoints[i][minIdx + 1]);
      if (minIdx > 0)
        candidates.push_back(contourPoints[i][minIdx - 1]);
    }

    // set up projection matrix, model view matrix is identity
    double fx = reconParams.imgWidth / 2.0;
    double fy = reconParams.imgHeight / 2.0;
    PhGUtils::Matrix4x4d Pmat(modelParams.camparams.fx / fx, 0, 0, 0,
      0, modelParams.camparams.fx / fy, 0, 0,
      0, 0, -100.0001 / 99.9999, -0.02 / 99.9999,
      0, 0, 1.0, 0);  // note here we need 1.0 instead of -1.0 for OpenGL

    // viewport transformation parameters
    int viewport[4] = { 0, 0, reconParams.imgWidth, reconParams.imgHeight };

    auto &tmesh = reconParams.tmesh;

    // project all candidates to image plane, find the new correspondence using nearest neighbor
    int contourPointsCount = 15;
    for (int i = 0; i < contourPointsCount; ++i) {
      int u = reconParams.cons[i].q.x;
      int v = reconParams.cons[i].q.y;

      // check the candidates
      float closestDist = numeric_limits<float>::max();
      int closestIdx = reconParams.cons[i].vidx;

      for (auto cidx : candidates) {
        int cOffset = cidx * 3;
        PhGUtils::Point3f fpt(tmesh(cOffset), tmesh(cOffset + 1), tmesh(cOffset + 2));
        double uf, vf, df;
        PhGUtils::projectPoint<double>(uf, vf, df, fpt.x, fpt.y, fpt.z, PhGUtils::Matrix4x4d::identity(), Pmat, viewport);
        vf = reconParams.imgHeight - 1 - vf;
        float du = u - uf;
        float dv = v - vf;
        float dist = du*du + dv*dv;
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = cidx;
        }
      }

      // update the constraint
      //cout << candidates.size() << "\t" << cons[i].vidx << " -> " << closestIdx << endl;
      reconParams.cons[i].vidx = closestIdx;
    }
  }
  int constraintCount() const {
    return reconParams.cons.size();
  }
  int constraintCountWithPrior() const {
    return reconParams.cons.size() + 1;
  }
  int getNDims_rt() {
    return MultilinearModelParameters::nRTparams;
  }
  int getNDims_id() {
    return modelParams.Wid.length();
  }
  int getNDims_exp() {
    return modelParams.Wexp.length();
  }
  void setConstraints(const vector<Constraint_2D> &c) {
    reconParams.cons = c;
  }
  void updateConstraints(const vector<Constraint_2D> &c) {
    for (int i = 0; i < reconParams.cons.size(); ++i) {
      reconParams.cons[i].q = c[i].q;
    }
  }
  vector<int> getConstraintIndices() {
    vector<int> indices(reconParams.cons.size());
    for (int i = 0; i < reconParams.cons.size(); ++i) {
      indices[i] = reconParams.cons[i].vidx;
    }
    return indices;
  }

  void updateCameraParameters() {
    int npts = reconParams.cons.size();
    /// since only 2D feature points are used
    /// compute the focal length analytically
    auto &tmc = reconParams.model_projected.tm;
    double numer = 0.0, denom = 0.0;
    double p0 = modelParams.camparams.cx, q0 = modelParams.camparams.cy;
    for (int i = 0, vidx = 0; i < npts; ++i) {
      double px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      double pi = reconParams.cons[i].q.x, qi = reconParams.cons[i].q.y;
      PhGUtils::transformPoint(px, py, pz, modelParams.Rmat, modelParams.Tvec);
      double xz = px / pz;
      double yz = py / pz;
      numer += yz * (q0 - qi) - xz*(p0 - pi);
      denom += (yz*yz + xz*xz);
    }
    double nf = -numer / denom;
    //cout << nf << endl;
    modelParams.camparams.fx = -nf;
    modelParams.camparams.fy = nf;
  }
  
  void updateTM0() {
    reconParams.model_projected.updateTM0(modelParams.Wid);
  }
  void updateTM1() {
    reconParams.model_projected.updateTM1(modelParams.Wexp);
  }
  void transformTM0() {
    reconParams.model_projected.transformTM0(modelParams.Rmat);
  }
  void transformTM1() {
    reconParams.model_projected.transformTM1(modelParams.Rmat);
  }
  void updateTMWithMode1() {
    reconParams.model_projected.updateTMWithMode1(modelParams.Wid);
  }
  void updateTMWithMode0() {
    reconParams.model_projected.updateTMWithMode0(modelParams.Wexp);
  }
  void initializeRigidTransformation() {
    auto &params = modelParams;
    // estimate a position
    // no rotation initially
    params.RTparams[0] = 0; params.RTparams[1] = 0; params.RTparams[2] = 0;
    // at origin initially
    params.RTparams[3] = -0.05; params.RTparams[4] = -0.04; params.RTparams[5] = -1.0;

    params.R = arma::mat(3, 3);
    params.R(0, 0) = 1.0, params.R(1, 1) = 1.0, params.R(2, 2) = 1.0;
    params.T = arma::vec(3);

    params.Rmat = PhGUtils::Matrix3x3d::identity();
    params.Tvec = PhGUtils::Point3d::zero();
  }

  PhGUtils::Matrix3x3d getProjectionMatrix() {
    return PhGUtils::Matrix3x3d(modelParams.camparams.fx, 0, modelParams.camparams.cx,
      0, modelParams.camparams.fy, modelParams.camparams.cy,
      0, 0, 1);
  }

  void setImageSize(int w, int h) {
    modelParams.camparams.cx = w / 2.0;
    modelParams.camparams.cy = h / 2.0;
    modelParams.camparams.fx = -1000.0;
    modelParams.camparams.fy = 1000.0;

    reconParams.imgWidth = w;
    reconParams.imgHeight = h;
  }

  void updateProjectedTensor(const MultilinearModel<double> &m) {
    reconParams.model_projected = m.project(getConstraintIndices());
    reconParams.model_projected.applyWeights(modelParams.Wid, modelParams.Wexp);

    // assume no rotation at this point
    reconParams.model_projected.tm0RT = reconParams.model_projected.tm0;
    reconParams.model_projected.tm1RT = reconParams.model_projected.tm1;
  }

  double* getRTParams() {
    return modelParams.RTparams;
  }

  double updateRTParams() {
    modelParams.Rmat = PhGUtils::rotationMatrix(modelParams.RTparams[0], modelParams.RTparams[1], modelParams.RTparams[2]);
    double diff = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        diff = max(diff, fabs(modelParams.R(i, j) - modelParams.Rmat(i, j)));
        modelParams.R(i, j) = modelParams.Rmat(i, j);
      }
    }

    modelParams.Tvec.x = modelParams.RTparams[3], modelParams.Tvec.y = modelParams.RTparams[4], modelParams.Tvec.z = modelParams.RTparams[5];
    diff = max(diff, fabs(modelParams.Tvec.x - modelParams.T(0)));
    diff = max(diff, fabs(modelParams.Tvec.y - modelParams.T(1)));
    diff = max(diff, fabs(modelParams.Tvec.z - modelParams.T(2)));
    modelParams.T(0) = modelParams.RTparams[3], modelParams.T(1) = modelParams.RTparams[4], modelParams.T(2) = modelParams.RTparams[5];

    return diff;
  }

  double* getWidParams() {
    return modelParams.Wid.rawptr();
  }

  double updateWidParams(const vector<double> &wid, double underrelax_factor) {
    double diff = 0.0;
    //cout << "Wid_before = ";
    //for (int i = 0; i < wid.size(); ++i) {
    //  cout << modelParams.Wid(i) << " ";
    //}
    //cout << endl;;

    //cout << "Wid = ";
    for (int i = 0; i < wid.size(); i++) {
      diff = max(diff, fabs(modelParams.Wid(i) - wid[i]) / (fabs(modelParams.Wid(i)) + 1e-12f));
      modelParams.Wid(i) = underrelax_factor * wid[i] + (1.0 - underrelax_factor) * modelParams.Wid(i);
      //cout << modelParams.Wid(i) << ' ';
    }
    //cout << endl;

    return diff;
  }

  double* getWexpParams() {
    return modelParams.Wexp.rawptr();
  }

  double updateWexpParams(const vector<double> &wexp, double underrelax_factor) {
    double diff = 0;
    for (int i = 0; i < wexp.size(); i++) {
      diff = max(diff, fabs(modelParams.Wexp(i) - wexp[i]) / (fabs(modelParams.Wexp(i)) + 1e-12));
      modelParams.Wexp(i) = underrelax_factor * wexp[i] + (1.0 - underrelax_factor)*modelParams.Wexp(i);
      //cout << wexp[i] << ' ';
    }
    return diff;
  }
};

struct MultiImageParameters {
  MultiImageParameters() {
    constraintsSet = false;
    fitExpression = false;
  }

  bool constraintsSet;
  bool fitExpression;
  size_t nImages;
  
  // weight for each image
  vector<double> wImg;

  MultilinearModelPriorParameters priorParams;
  vector<MultilinearModelParameters> modelParams;
  vector<ReconstructionParameters> reconParams;

  ConvergenceParameters convParams;

  void loadPrior(const string &filename_id, const string &filename_exp) {
    priorParams.load(filename_id, filename_exp);
  }

  void init(int n = 0) {
    nImages = 0;

    if (n == 0) return;

    PhGUtils::message("initializing weights ...");
    nImages = n;

    modelParams.resize(nImages);
    reconParams.resize(nImages);

    // dimension of the identity weights
    int ndims_id = priorParams.mu_wid0.n_elem;
    // dimension of the expression weights
    int ndims_exp = priorParams.mu_wexp0.n_elem;

    RTparams.resize(MultilinearModelParameters::nRTparams * nImages);
    Wid.resize(ndims_id);
    Wexp.resize(nImages * ndims_exp);

    // use the first person initially
    for (int i = 0; i < ndims_id; i++) {
      Wid[i] = priorParams.mu_wid0(i);
    }

    // use neutral face initially
    for (int i = 0; i < ndims_exp; i++) {
      Wexp[i] = priorParams.mu_wexp0(i);
    }

    for (auto &mp : modelParams) {
      mp.Wid.resize(ndims_id);
      mp.Wexp.resize(ndims_exp);

      // use the first person initially
      for (int i = 0; i < ndims_id; i++) {
        mp.Wid(i) = priorParams.mu_wid0(i);
      }

      // use neutral face initially
      for (int i = 0; i < ndims_exp; i++) {
        mp.Wexp(i) = priorParams.mu_wexp0(i);
      }
    }  

    PhGUtils::message("done.");
  }
  bool isConstraintsSet() {
    return constraintsSet;
  }
  void setConstraints(const vector<vector<Constraint_2D>> &constraints) {
    assert(nImages == constraints.size());

    for (int i = 0; i < nImages; ++i) {
      reconParams[i].cons = constraints[i];
    }

    constraintsSet = true;
  }
  void updateConstraints(const vector<vector<Constraint_2D>> &constraints) {
    assert(nImages == constraints.size());

    for (int i = 0; i < nImages; ++i) {
      for (int j = 0; j < constraints[i].size(); ++j) {
        reconParams[i].cons[j].q = constraints[i][j].q;
      }      
    }
  }
  void updateMesh(MultilinearModel<double> &m) {
    for (int i = 0; i < nImages; ++i) {
      m.applyWeights(modelParams[i].Wid, modelParams[i].Wexp);
      reconParams[i].tmesh = m.tm;
    }
  }
  void setBaseMesh(const string &filename){
    PhGUtils::OBJLoader loader;
    loader.load(filename);
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].baseMesh.initWithLoader(loader);
    }
    cout << "base mesh set." << endl;
  }
  vector<Tensor1<double>> getFittedMeshes() {
    vector<Tensor1<double>> res(nImages);
    for (int i = 0; i < nImages; ++i) {
      res[i] = reconParams[i].tmesh;
    }
    return res;
  }

  PhGUtils::QuadMesh& getMesh(int idx) {
    return reconParams[idx].baseMesh;
  }

  float getFocalLength(int idx) const {
    return modelParams[idx].camparams.fy;
  }

  void getTranslation(int idx, float &x, float &y, float &z) {
    x = modelParams[idx].Tvec.x;
    y = modelParams[idx].Tvec.y;
    z = modelParams[idx].Tvec.z;
  }

  void initializeRigidTransformation() {
    for (int i = 0; i < nImages; ++i) {
      auto &params = modelParams[i];
      // estimate a position
      // no rotation initially
      params.RTparams[0] = 0; params.RTparams[1] = 0; params.RTparams[2] = 0;
      // at origin initially
      params.RTparams[3] = -0.05; params.RTparams[4] = -0.04; params.RTparams[5] = -1.0;

      params.R = arma::mat(3, 3);
      params.R(0, 0) = 1.0, params.R(1, 1) = 1.0, params.R(2, 2) = 1.0;
      params.T = arma::vec(3);

      params.Rmat = PhGUtils::Matrix3x3d::identity();
      params.Tvec = PhGUtils::Point3d::zero();
    }
  }

  void setImageSize(const vector<PhGUtils::Point2i> &sizes) {
    for (int i = 0; i < nImages; ++i) {
      int w = sizes[i].x, h = sizes[i].y;

      modelParams[i].camparams.cx = w / 2.0;
      modelParams[i].camparams.cy = h / 2.0;
      modelParams[i].camparams.fx = -1000.0;
      modelParams[i].camparams.fy = 1000.0;

      reconParams[i].imgWidth = w;
      reconParams[i].imgHeight = h;
    }
  }

  void generateFittedMesh(MultilinearModel<double> &m, bool applyWeights = true) {

    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      if (applyWeights) {
        //modelParams[imgidx].Wid.print("Wid");
        m.updateTM0(modelParams[imgidx].Wid);
        m.updateTM1(modelParams[imgidx].Wexp);
        m.updateTMWithMode0(modelParams[imgidx].Wexp);
      }

      auto &tplt = m.tm;
      int nverts = tplt.length() / 3;
      arma::mat pt(3, nverts);
      for (int i = 0, idx = 0; i < nverts; i++, idx += 3) {
        pt(0, i) = tplt(idx);
        pt(1, i) = tplt(idx + 1);
        pt(2, i) = tplt(idx + 2);
      }

      auto &tmesh = reconParams[imgidx].tmesh;
      tmesh.resize(tplt.length());
      // batch rotation processing
      arma::mat pt_trans = modelParams[imgidx].R * pt;
#pragma omp parallel for
      for (int i = 0; i < nverts; i++) {
        int idx = i * 3;
        tmesh(idx) = pt_trans(0, i) + modelParams[imgidx].T(0);
        tmesh(idx + 1) = pt_trans(1, i) + modelParams[imgidx].T(1);
        tmesh(idx + 2) = pt_trans(2, i) + modelParams[imgidx].T(2);
      }

      tmesh.write("0.mesh");

      auto &baseMesh = reconParams[imgidx].baseMesh;
      for (int i = 0; i < tmesh.length() / 3; i++) {
        int idx = i * 3;
        baseMesh.vertex(i).x = tmesh(idx++);
        baseMesh.vertex(i).y = tmesh(idx++);
        baseMesh.vertex(i).z = tmesh(idx);
      }
    }
  }

  void updateTM0() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.updateTM0(modelParams[i].Wid);
    }
  }
  void updateTM1() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.updateTM1(modelParams[i].Wexp);
    }
  }
  void transformTM0() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.transformTM0(modelParams[i].Rmat);
    }
  }
  void transformTM1() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.transformTM1(modelParams[i].Rmat);
    }
  }
  void updateTMWithMode1() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.updateTMWithMode1(modelParams[i].Wid);
    }
  }
  void updateTMWithMode0() {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected.updateTMWithMode0(modelParams[i].Wexp);
    }
  }
  vector<int> getConstraintIndices(int idx) {
    vector<int> indices(reconParams[idx].cons.size());
    for (int i = 0; i < reconParams[idx].cons.size(); ++i) {
      indices[i] = reconParams[idx].cons[i].vidx;
    }
    //cout << "constraint size = " << indices.size() << endl;
    return indices;
  }
  void updateProjectedTensor(const MultilinearModel<double> &m) {
    for (int i = 0; i < nImages; ++i) {
      reconParams[i].model_projected = m.project(getConstraintIndices(i));
      reconParams[i].model_projected.applyWeights(modelParams[i].Wid, modelParams[i].Wexp);

      // assume no rotation at this point
      reconParams[i].model_projected.tm0RT = reconParams[i].model_projected.tm0;
      reconParams[i].model_projected.tm1RT = reconParams[i].model_projected.tm1;
    }
  }
  void updateCameraParameters() {
    for (int imgIdx = 0; imgIdx < nImages; ++imgIdx) {
      int npts = reconParams[imgIdx].cons.size();
      /// since only 2D feature points are used
      /// compute the focal length analytically
      auto &tmc = reconParams[imgIdx].model_projected.tm;
      double numer = 0.0, denom = 0.0;
      double p0 = modelParams[imgIdx].camparams.cx, q0 = modelParams[imgIdx].camparams.cy;
      for (int i = 0, vidx = 0; i < npts; ++i) {
        double px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
        double pi = reconParams[imgIdx].cons[i].q.x, qi = reconParams[imgIdx].cons[i].q.y;
        PhGUtils::transformPoint(px, py, pz, modelParams[imgIdx].Rmat, modelParams[imgIdx].Tvec);
        double xz = px / pz;
        double yz = py / pz;
        numer += yz * (q0 - qi) - xz*(p0 - pi);
        denom += (yz*yz + xz*xz);
      }
      double nf = -numer / denom;
      //cout << nf << endl;
      modelParams[imgIdx].camparams.fx = -nf;
      modelParams[imgIdx].camparams.fy = nf;
    }
  }

  vector<double> RTparams;
  vector<double> Wid;
  vector<double> Wexp;

  int constraintCount() {
    return nImages * reconParams.front().cons.size();
  }
  int constraintCountWithPrior() const {
    return nImages * (reconParams.front().cons.size() + 1);
  }
  int getNDims_rt() {
    return nImages * MultilinearModelParameters::nRTparams;
  }
  int getNDims_id() {
    return priorParams.mu_wid0.n_elem;  // only one identity!!!
  }
  int getNDims_exp() {
    return nImages * priorParams.mu_wexp0.n_elem;
  }

  double* getRTParams() {
    // collect rigid transform parameters
    for (int i = 0; i < nImages; ++i) {
      for (int j = 0; j < MultilinearModelParameters::nRTparams; ++j) {
        RTparams[i*MultilinearModelParameters::nRTparams + j] = modelParams[i].RTparams[j];
      }
    }
    return &(RTparams[0]);
  }

  double updateRTParams() {
    for (int i = 0; i < nImages; ++i) {
      for (int j = 0; j < MultilinearModelParameters::nRTparams; ++j) {
        modelParams[i].RTparams[j] = RTparams[i*MultilinearModelParameters::nRTparams + j];
      }
    }

    double diff = 0;
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      modelParams[imgidx].Rmat = PhGUtils::rotationMatrix(modelParams[imgidx].RTparams[0], modelParams[imgidx].RTparams[1], modelParams[imgidx].RTparams[2]);
      
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          diff = max(diff, fabs(modelParams[imgidx].R(i, j) - modelParams[imgidx].Rmat(i, j)));
          modelParams[imgidx].R(i, j) = modelParams[imgidx].Rmat(i, j);
        }
      }

      modelParams[imgidx].Tvec.x = modelParams[imgidx].RTparams[3], modelParams[imgidx].Tvec.y = modelParams[imgidx].RTparams[4], modelParams[imgidx].Tvec.z = modelParams[imgidx].RTparams[5];
      diff = max(diff, fabs(modelParams[imgidx].Tvec.x - modelParams[imgidx].T(0)));
      diff = max(diff, fabs(modelParams[imgidx].Tvec.y - modelParams[imgidx].T(1)));
      diff = max(diff, fabs(modelParams[imgidx].Tvec.z - modelParams[imgidx].T(2)));
      modelParams[imgidx].T(0) = modelParams[imgidx].RTparams[3], modelParams[imgidx].T(1) = modelParams[imgidx].RTparams[4], modelParams[imgidx].T(2) = modelParams[imgidx].RTparams[5];
    }
    return diff;
  }

  double* getWidParams() {
    return &(Wid[0]);
  }

  double updateWidParams(const vector<double> &wid, double underrelax_factor) {
    //cout << "Wid_before = ";
    for (int i = 0; i < Wid.size(); ++i) {
    //  cout << Wid[i] << " ";
      Wid[i] = wid[i] * underrelax_factor + (1.0 - underrelax_factor) * Wid[i];
    }
    //cout << endl;

    double diff = 0.0;
    int nparams = priorParams.mu_wid0.n_elem;
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      //cout << "Wid[" << imgidx << "] = ";
      for (int i = 0; i < nparams; i++) {
        diff = max(diff, fabs(modelParams[imgidx].Wid(i) - wid[i]) / (fabs(modelParams[imgidx].Wid(i)) + 1e-12f));
        modelParams[imgidx].Wid(i) = Wid[i];
        //cout << modelParams[imgidx].Wid(i) << ' ';
      }
      //cout << endl;
      //system("pause");
    }   

    return diff;
  }

  double* getWexpParams() {
    return &(Wexp[0]);
  }

  double updateWexpParams(const vector<double> &wexp, double underrelax_factor) {
    for (int i = 0; i < Wexp.size(); ++i) {
      Wexp[i] = wexp[i] * underrelax_factor + (1.0 - underrelax_factor) * Wexp[i];
    }

    double diff = 0;
    int nparams = priorParams.mu_wexp0.n_elem;
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      int offset = imgidx * nparams;
      for (int i = 0; i < nparams; i++) {
        diff = max(diff, fabs(modelParams[imgidx].Wexp(i) - wexp[offset+i]) / (fabs(modelParams[imgidx].Wexp(i)) + 1e-12));
        modelParams[imgidx].Wexp(i) = underrelax_factor * wexp[offset + i] + (1.0 - underrelax_factor)*modelParams[imgidx].Wexp(i);
        //cout << wexp[i] << ' ';
      }
    }
    return diff;
  }

  void updateContourPoints(const vector<vector<int>> &contourPoints,
    const vector<int> &contourVerts) {
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      auto &baseMesh = reconParams[imgidx].baseMesh;
      baseMesh.computeNormals(contourVerts);

      PhGUtils::Vector3f viewVec(0, 0, -1);

      // for each row, pick the best candidates
      int nContourRows = contourPoints.size();
      vector<int> candidates;
      for (int i = 0; i < nContourRows; ++i) {
        int minIdx = 0;
        float minProd = fabs(baseMesh.normal(contourPoints[i].front()).dot(viewVec));
        for (int j = 1; j < contourPoints[i].size(); ++j) {
          auto &nij = baseMesh.normal(contourPoints[i][j]);
          auto &vij = baseMesh.vertex(contourPoints[i][j]);
          viewVec = PhGUtils::Vector3f(
            vij.x,
            vij.y,
            vij.z
            ).normalized();
          float prod = fabs(nij.dot(viewVec));
          if (prod < minProd) {
            minIdx = j;
            minProd = prod;
          }
        }

        candidates.push_back(contourPoints[i][minIdx]);
        if (minIdx < contourPoints[i].size() - 1)
          candidates.push_back(contourPoints[i][minIdx + 1]);
        if (minIdx > 0)
          candidates.push_back(contourPoints[i][minIdx - 1]);
      }

      // set up projection matrix, model view matrix is identity
      double fx = reconParams[imgidx].imgWidth / 2.0;
      double fy = reconParams[imgidx].imgHeight / 2.0;
      PhGUtils::Matrix4x4d Pmat(modelParams[imgidx].camparams.fx / fx, 0, 0, 0,
        0, modelParams[imgidx].camparams.fx / fy, 0, 0,
        0, 0, -100.0001 / 99.9999, -0.02 / 99.9999,
        0, 0, 1.0, 0);  // note here we need 1.0 instead of -1.0 for OpenGL

      // viewport transformation parameters
      int viewport[4] = { 0, 0, reconParams[imgidx].imgWidth, reconParams[imgidx].imgHeight };

      auto &tmesh = reconParams[imgidx].tmesh;

      // project all candidates to image plane, find the new correspondence using nearest neighbor
      int contourPointsCount = 15;
      for (int i = 0; i < contourPointsCount; ++i) {
        int u = reconParams[imgidx].cons[i].q.x;
        int v = reconParams[imgidx].cons[i].q.y;

        // check the candidates
        float closestDist = numeric_limits<float>::max();
        int closestIdx = reconParams[imgidx].cons[i].vidx;

        for (auto cidx : candidates) {
          int cOffset = cidx * 3;
          PhGUtils::Point3f fpt(tmesh(cOffset), tmesh(cOffset + 1), tmesh(cOffset + 2));
          double uf, vf, df;
          PhGUtils::projectPoint<double>(uf, vf, df, fpt.x, fpt.y, fpt.z, PhGUtils::Matrix4x4d::identity(), Pmat, viewport);
          vf = reconParams[imgidx].imgHeight - 1 - vf;
          float du = u - uf;
          float dv = v - vf;
          float dist = du*du + dv*dv;
          if (dist < closestDist) {
            closestDist = dist;
            closestIdx = cidx;
          }
        }

        // update the constraint
        //cout << candidates.size() << "\t" << cons[i].vidx << " -> " << closestIdx << endl;
        reconParams[imgidx].cons[i].vidx = closestIdx;
      }
    }    
  }

  PhGUtils::Matrix3x3d getProjectionMatrix(int idx) {
    return PhGUtils::Matrix3x3d(modelParams[idx].camparams.fx, 0, modelParams[idx].camparams.cx,
      0, modelParams[idx].camparams.fy, modelParams[idx].camparams.cy,
      0, 0, 1);
  }

  vector<double> getFocalLength() {
    vector<double> fs;
    for (int i = 0; i < nImages; ++i) {
      fs.push_back(modelParams[i].camparams.fy);
    }
    return fs;
  }
};

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

struct DefaultParameters {
  DefaultParameters() {
    w_prior_id = 0.25;
    w_prior_exp = 0.25;
  }

  // weights prior
  // average identity and neutral expression
  arma::vec mu_wid0, mu_wexp0;
  // mean wid and mean wexp
  arma::vec mu_wid_orig, mu_wexp_orig;
  // mean wid and mean wexp, multiplied by the inverse of sigma_?
  arma::vec mu_wid, mu_wexp;
  // actually inv(sigma_wid), inv(sigma_wexp), same for below
  arma::mat sigma_wid, sigma_wexp;

  // parameters to optimize
  static const int nRTparams = 6;

  Tensor1<double> Wid, Wexp;
  double RTparams[nRTparams];
  arma::mat R;
  arma::vec T;
  PhGUtils::Matrix3x3d Rmat;
  PhGUtils::Point3d Tvec;

  CameraParams camparams;

  int imgWidth, imgHeight;
  Tensor1<double> tmesh;
  shared_ptr<PhGUtils::QuadMesh> baseMesh;
  vector<Constraint_2D> cons;

  MultilinearModel<double> model_projected;

  // convergence criteria
  struct ConvergenceParams {
    // not used yet
  };

  bool fitExpression;
  float w_prior_id, w_prior_exp;

  void loadPrior(const string &filename_id, const string &filename_exp) {
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
  void init() {
    PhGUtils::message("initializing weights ...");
    Wid.resize(mu_wid0.n_elem);
    Wexp.resize(mu_wexp0.n_elem);

    // use the first person initially
    for (int i = 0; i < Wid.length(); i++) {
      Wid(i) = mu_wid0(i);
    }

    // use neutral face initially
    for (int i = 0; i < Wexp.length(); i++) {
      Wexp(i) = mu_wexp0(i);
    }

    PhGUtils::message("done.");
  }
  void setBaseMesh(PhGUtils::QuadMesh &m) {
    baseMesh.reset(&m);
  }
  void updateMesh(MultilinearModel<double> &m) {
    m.applyWeights(Wid, Wexp);
    tmesh = m.tm;
  }
  void generateFittedMesh(const MultilinearModel<double> &m) {
    auto &tplt = m.tm;
    int nverts = tplt.length() / 3;
    arma::mat pt(3, nverts);
    for (int i = 0, idx = 0; i < nverts; i++, idx += 3) {
      pt(0, i) = tplt(idx);
      pt(1, i) = tplt(idx + 1);
      pt(2, i) = tplt(idx + 2);
    }

    // batch rotation processing
    arma::mat pt_trans = R * pt;
#pragma omp parallel for
    for (int i = 0; i < nverts; i++) {
      int idx = i * 3;
      tmesh(idx) = pt_trans(0, i) + T(0);
      tmesh(idx + 1) = pt_trans(1, i) + T(1);
      tmesh(idx + 2) = pt_trans(2, i) + T(2);
    }

    for (int i = 0; i < tmesh.length() / 3; i++) {
      int idx = i * 3;
      baseMesh->vertex(i).x = tmesh(idx++);
      baseMesh->vertex(i).y = tmesh(idx++);
      baseMesh->vertex(i).z = tmesh(idx);
    }
  }
  void updateContourPoints(const vector<vector<int>> &contourPoints,
    const vector<int> &contourVerts) {
    baseMesh->computeNormals(contourVerts);

    PhGUtils::Vector3f viewVec(0, 0, -1);

    // for each row, pick the best candidates
    int nContourRows = contourPoints.size();
    vector<int> candidates;
    for (int i = 0; i < nContourRows; ++i) {
      int minIdx = 0;
      float minProd = fabs(baseMesh->normal(contourPoints[i].front()).dot(viewVec));
      for (int j = 1; j < contourPoints[i].size(); ++j) {
        auto &nij = baseMesh->normal(contourPoints[i][j]);
        auto &vij = baseMesh->vertex(contourPoints[i][j]);
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
    double fx = imgWidth / 2.0;
    double fy = imgHeight / 2.0;
    PhGUtils::Matrix4x4d Pmat(camparams.fx / fx, 0, 0, 0,
      0, camparams.fx / fy, 0, 0,
      0, 0, -100.0001 / 99.9999, -0.02 / 99.9999,
      0, 0, 1.0, 0);  // note here we need 1.0 instead of -1.0 for OpenGL

    // viewport transformation parameters
    int viewport[4] = { 0, 0, imgWidth, imgHeight };

    // project all candidates to image plane, find the new correspondence using nearest neighbor
    int contourPointsCount = 15;
    for (int i = 0; i < contourPointsCount; ++i) {
      int u = cons[i].q.x;
      int v = cons[i].q.y;

      // check the candidates
      float closestDist = numeric_limits<float>::max();
      int closestIdx = cons[i].vidx;

      for (auto cidx : candidates) {
        int cOffset = cidx * 3;
        PhGUtils::Point3f fpt(tmesh(cOffset), tmesh(cOffset + 1), tmesh(cOffset + 2));
        double uf, vf, df;
        PhGUtils::projectPoint<double>(uf, vf, df, fpt.x, fpt.y, fpt.z, PhGUtils::Matrix4x4d::identity(), Pmat, viewport);
        vf = imgHeight - 1 - vf;
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
      cons[i].vidx = closestIdx;
    }
  }
  int constraintCount() const {
    return cons.size();
  }
  void setConstraints(const vector<Constraint_2D> &c) {
    cons = c;
  }
  void updateConstraints(const vector<Constraint_2D> &c) {
    for (int i = 0; i < cons.size(); ++i) {
      cons[i].q = c[i].q;
    }
  }
  vector<int> getConstraintIndices() {
    vector<int> indices(cons.size());
    for (int i = 0; i < cons.size(); ++i) {
      indices[i] = cons[i].vidx;
    }
    return indices;
  }

  void updateCameraParameters(const MultilinearModel<double> &m) {
    int npts = cons.size();
    /// since only 2D feature points are used
    /// compute the focal length analytically
    auto &tmc = m.tm;
    double numer = 0.0, denom = 0.0;
    double p0 = camparams.cx, q0 = camparams.cy;
    for (int i = 0, vidx = 0; i < npts; ++i) {
      double px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      double pi = cons[i].q.x, qi = cons[i].q.y;
      PhGUtils::transformPoint(px, py, pz, Rmat, Tvec);
      double xz = px / pz;
      double yz = py / pz;
      numer += yz * (q0 - qi) - xz*(p0 - pi);
      denom += (yz*yz + xz*xz);
    }
    double nf = -numer / denom;
    //cout << nf << endl;
    camparams.fx = -nf;
    camparams.fy = nf;
  }
};

struct MultiImageParameters {
  size_t nImages;
  
  // weight for each image
  vector<double> wImg;

  // weights prior
  // average identity and neutral expression
  arma::vec mu_wid0, mu_wexp0;
  // mean wid and mean wexp
  arma::vec mu_wid_orig, mu_wexp_orig;
  // mean wid and mean wexp, multiplied by the inverse of sigma_?
  arma::vec mu_wid, mu_wexp;
  // actually inv(sigma_wid), inv(sigma_wexp), same for below
  arma::mat sigma_wid, sigma_wexp;

  // parameters to optimize
  static const int nRTparams = 6;

  Tensor1<double> Wid;
  vector<Tensor1<double>> Wexp;
  vector<vector<double>> RTparams;
  vector<arma::mat> R;
  vector<arma::vec> T;
  vector<PhGUtils::Matrix3x3d> Rmat;
  vector<PhGUtils::Point3d> Tvec;

  vector<CameraParams> camparams;

  vector<int> imgWidth, imgHeight;
  vector<Tensor1<double>> tmesh;
  vector<PhGUtils::QuadMesh> baseMesh;
  vector<vector<Constraint_2D>> cons;
  vector<vector<int>> contourVerts;
  
  // convergence criteria
  struct ConvergenceParams {
    // not used yet
  };

  bool fitExpression;
  float w_prior_id, w_prior_exp;

  void loadPrior(const string &filename_id, const string &filename_exp) {
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
  void init() {
    PhGUtils::message("initializing weights ...");
    Wid.resize(mu_wid0.n_elem);

    for (auto &w : Wexp) {
      w.resize(mu_wexp.n_elem);
    }

    // use the first person initially
    for (int i = 0; i < Wid.length(); i++) {
      Wid(i) = mu_wid0(i);
    }

    // use neutral face initially
    for (auto &w : Wexp) {
      for (int i = 0; i < w.length(); i++) {
        w(i) = mu_wexp0(i);
      }
    }

    PhGUtils::message("done.");
  }
  void updateMesh(const MultilinearModel<double> &m) {
  }
  void setBaseMesh(PhGUtils::QuadMesh &m){

  }
};

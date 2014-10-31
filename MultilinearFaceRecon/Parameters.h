#pragma once

#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include "Math/denseblas.h"
#include "Utils/utility.hpp"

struct DefaultParameters {
  DefaultParameters() {
    w_prior_id = 1.0;
    w_prior_exp = 1.0;
  }

  // weights prior
  // average identity and neutral expression
  arma::fvec mu_wid0, mu_wexp0;
  // mean wid and mean wexp
  arma::fvec mu_wid_orig, mu_wexp_orig;
  // mean wid and mean wexp, multiplied by the inverse of sigma_?
  arma::fvec mu_wid, mu_wexp;
  // actually inv(sigma_wid), inv(sigma_wexp), same for below
  arma::fmat sigma_wid, sigma_wexp;

  // parameters to optimize
  static const int nRTparams = 6;

  Tensor1<float> Wid, Wexp;
  float RTparams[nRTparams];
  arma::fmat R;
  arma::fvec T;
  PhGUtils::Matrix3x3f Rmat;
  PhGUtils::Point3f Tvec;

  struct CameraParams {
    CameraParams() :fx(-525.0), fy(525.0), cx(320.0), cy(240.0){}
    double fx, fy, cx, cy;  /// estimated projection matrix
  } camparams;

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

    fwid.read(reinterpret_cast<char*>(mu_wid0.memptr()), sizeof(float)*ndims);
    fwid.read(reinterpret_cast<char*>(mu_wid.memptr()), sizeof(float)*ndims);
    fwid.read(reinterpret_cast<char*>(sigma_wid.memptr()), sizeof(float)*ndims*ndims);

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

    fwexp.read(reinterpret_cast<char*>(mu_wexp0.memptr()), sizeof(float)*ndims);
    fwexp.read(reinterpret_cast<char*>(mu_wexp.memptr()), sizeof(float)*ndims);
    fwexp.read(reinterpret_cast<char*>(sigma_wexp.memptr()), sizeof(float)*ndims*ndims);

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
};


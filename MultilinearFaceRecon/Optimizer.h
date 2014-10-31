#pragma once

#include "Math/Tensor.hpp"
#include "MultilinearModel.h"
#include "PointConstraint.h"
#include "EnergyFunctions.h"

enum FittingOption {
  FIT_POSE = 0,
  FIT_IDENTITY,
  FIT_EXPRESSION,
  FIT_POSE_AND_IDENTITY,
  FIT_POSE_AND_EXPRESSION,
  FIT_ALL
};

// wrappers
template <typename T>
void cost_func_pose(float *p, float *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_pose(p, hx, m, n, NULL);
}

template <typename T>
void jac_func_pose(float *p, float *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_pose(p, J, m, n, NULL);
}

template <typename T>
void cost_func_identity(float *p, float *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_identity(p, hx, m, n, NULL);
}

template <typename T>
void jac_func_identity(float *p, float *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_identity(p, J, m, n, NULL);
}

template <typename T>
void cost_func_expression(float *p, float *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_expression(p, hx, m, n, NULL);
}

template <typename T>
void jac_func_expression(float *p, float *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_expression(p, J, m, n, NULL);
}

template <typename Constraint, typename Parameters, typename EnergyFunction>
class Optimizer {
public:
  Optimizer(){}
  Optimizer(MultilinearModel<float> &model, Parameters &params) :model(model), params(params){
    // initialize the mesh tensor
    tmesh = model.tm;
  }
  ~Optimizer(){}

  Tensor1<float> fit(const vector<Constraint> &constraints,
                     FittingOption &ops) {

    // check if we need to update the projected tensor
    checkAndUpdateConstraints(constraints);

    switch (ops) {
    case FIT_POSE:
      fit_pose();
      break;
    case FIT_IDENTITY:
      fit_identity();
      break;
    case FIT_EXPRESSION:
      fit_expression();
      break;
    case FIT_POSE_AND_EXPRESSION:
      fit_pose_and_expression();
      break;
    case FIT_POSE_AND_IDENTITY:
      fit_pose_and_identity();
      break;
    case FIT_ALL:
      fit_all();
      break;
    }
    generateFittedMesh();

    return tmesh;
  }

private:
  void checkAndUpdateConstraints(const vector<Constraint> &constraints) {
    bool needInitialize = false;
    if (constraints.size() != cons.size()) {
      // direct update
      needInitialize = true;
    }
    else{
      // check every single
      for (int i = 0; i < constraints.size(); ++i) {
        if (!cons[i].hasSameId(constraints[i])) {
          needInitialize = true;
          break;
        }
      }
    }
    cons = constraints;

    if (needInitialize) {
      updateProjectedTensors();
      initializeRigidTransformation();
      engfun.reset(new EnergyFunction(model_projected, params, cons));
    }      
  }

  void updateProjectedTensors() {
    vector<int> indices(cons.size());
    for (int i = 0; i < cons.size(); ++i) {
      indices[i] = cons[i].vidx;
    }
    model_projected = model.project(indices);
    cout << model_projected.core.dim(0) << ", " << model_projected.core.dim(1) << ", " << model_projected.core.dim(2) << endl;
    model_projected.applyWeights(params.Wid, params.Wexp);

    // assume no rotation at this point
    model_projected.tm0RT = model_projected.tm0;
    model_projected.tm1RT = model_projected.tm1;
  }

  void initializeRigidTransformation() {
    // estimate a position
    // no rotation initially
    params.RTparams[0] = 0; params.RTparams[1] = 0; params.RTparams[2] = 0;
    // at origin initially
    params.RTparams[3] = -0.05; params.RTparams[4] = -0.04; params.RTparams[5] = -0.95;

    params.R = arma::fmat(3, 3);
    params.R(0, 0) = 1.0, params.R(1, 1) = 1.0, params.R(2, 2) = 1.0;
    params.T = arma::fvec(3);

    params.Rmat = PhGUtils::Matrix3x3f::identity();
    params.Tvec = PhGUtils::Point3f::zero();
  }

  float computeError()
  {
    return engfun->error();
  }

  void fit_pose() {
    PhGUtils::message("fitting pose ...");
    fitRigidTransformation();
    float E = computeError();
    cout << "Error = " << E << endl;
  }

  void fit_identity(){

  }
  
  void fit_expression() {

  }

  void fit_pose_and_identity() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 32;
    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;

    while (!converged && iters++ < MAXITERS) {
      converged = true;

      timerRT.tic();
      converged &= fitRigidTransformation();
      timerRT.toc();

      // apply the new global transformation to tm1c
      // because tm1c is required in fitting identity weights
      timerOther.tic();
      model_projected.transformTM1(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights();
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      model_projected.updateTMWithMode1(params.Wid);
      timerOther.toc();

      /// optimize for camera parameters
      converged &= fitCameraParameters();

      timerOther.tic();
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      // uncomment to show the transformation process		

      /*
      transformMesh();
      Rmat.print("R");
      Tvec.print("T");
      emit oneiter();
      QApplication::processEvents();
      ::system("pause");
      */
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    model_projected.updateTM0(params.Wid);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.Wid);
    model.updateTMWithMode1(params.Wid);
    timerOther.toc();
  }

  void fit_pose_and_expression() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 32;
    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;

    while (!converged && iters++ < MAXITERS) {
      converged = true;

      timerRT.tic();
      converged &= fitRigidTransformation();
      timerRT.toc();

      //cout << "fitting expression weights ..." << endl;
      timerOther.tic();
      // apply the global transformation to tm0c
      // because tm0c is required in fitting expression weights
      model_projected.transformTM0(params.Rmat);
      timerOther.toc();

      timerExp.tic();
      converged &= fitExpressionWeights();
      timerExp.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      model_projected.updateTMWithMode0(params.Wexp);
      timerOther.toc();
      
      E = computeError();
      //PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      // uncomment to show the transformation process		

      /*
      transformMesh();
      Rmat.print("R");
      Tvec.print("T");
      emit oneiter();
      QApplication::processEvents();
      ::system("pause");
      */
    }

    timerTransform.tic();
    model.updateTMWithMode0(params.Wexp);
    timerTransform.toc();
  }

  void fit_all() {

  }

  bool fitCameraParameters() {
    int npts = cons.size();
    /// since only 2D feature points are used
    /// compute the focal length analytically
    auto &tmc = model_projected.tm;
    double numer = 0.0, denom = 0.0;
    double p0 = params.camparams.cx, q0 = params.camparams.cy;
    for (int i = 0, vidx = 0; i < npts; ++i) {
      float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      float pi = cons[i].q.x, qi = cons[i].q.y;
      PhGUtils::transformPoint(px, py, pz, params.Rmat, params.Tvec);
      float xz = px / pz;
      float yz = py / pz;
      numer += yz * (q0 - qi) - xz*(p0 - pi);
      denom += (yz*yz + xz*xz);
    }
    double nf = -numer / denom;
    cout << nf << endl;
    params.camparams.fx = -nf;
    params.camparams.fy = nf;
    return true;
  }

  bool fitRigidTransformation() {
    //cout << "fitting rigid transformation ..." << endl;
    int npts = cons.size();
    vector<float> meas(npts, 0.0);
    auto &RTparams = params.RTparams;
    int nparams = Parameters::nRTparams;

    float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9 };  // tau, eps1, eps2, eps3
    int iters = slevmar_der(cost_func_pose<EnergyFunction>, jac_func_pose<EnergyFunction>, 
      RTparams, &(meas[0]), nparams, npts, 128, opts, NULL, NULL, NULL, engfun.get());
    //cout << "iters = " << iters << endl;
    // set up the matrix and translation vector
    params.Rmat = PhGUtils::rotationMatrix(RTparams[0], RTparams[1], RTparams[2]);
    float diff = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        diff = max(diff, fabs(params.R(i, j) - params.Rmat(i, j)));
        params.R(i, j) = params.Rmat(i, j);
      }
    }

    params.Tvec.x = RTparams[3], params.Tvec.y = RTparams[4], params.Tvec.z = RTparams[5];
    diff = max(diff, fabs(params.Tvec.x - params.T(0)));
    diff = max(diff, fabs(params.Tvec.y - params.T(1)));
    diff = max(diff, fabs(params.Tvec.z - params.T(2)));
    params.T(0) = RTparams[3], params.T(1) = RTparams[4], params.T(2) = RTparams[5];

    /*
    cout << "after" << endl;
    cout << params.R << endl;
    cout << params.T << endl;
    */
    return diff < 2.5e-6;
  }

  bool fitIdentityWeights() {
    int nparams = model_projected.core.dim(0);
    vector<float> wid(params.Wid.rawptr(), params.Wid.rawptr() + nparams);
    int npts = cons.size();
    vector<float> meas(npts + 1, 0.0);
#if 0
    float opts[5] = { 1e-3, 1e-9, 1e-9, 1e-9, 1e-6 };
    int iters = slevmar_dif(cost_func_identity<EnergyFunction>, &(params[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, engfun.get());
#else
    // evaluate the jacobian
    /*
    vector<float> errorvec(npts + 1);
    slevmar_chkjac(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), nparams, npts + 1, engfun.get(), &(errorvec[0]));
    PhGUtils::message("error vec[identity]");
    PhGUtils::printArray(&(errorvec[0]), npts + 1);
    */

    float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9 };
    int iters = slevmar_der(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, engfun.get());
#endif
    //cout << "identity fitting finished in " << iters << " iterations." << endl;

    //debug("rtn", rtn);
    float diff = 0.0;
    //b.print("b");
    for (int i = 0; i < nparams; i++) {
      diff = max(diff, fabs(params.Wid(i) - wid[i]) / (fabs(params.Wid(i)) + 1e-12f));
      params.Wid(i) = wid[i];
      //cout << params[i] << ' ';
    }
    //cout << endl;

    cout << "diff = " << diff << endl;

    return diff < 1e-4;
  }

  bool fitExpressionWeights() {
    // fix both rotation and identity weights, solve for expression weights
    int nparams = model_projected.core.dim(1);
    vector<float> wexp(params.Wexp.rawptr(), params.Wexp.rawptr() + nparams);
    int npts = cons.size();
    vector<float> meas(npts + 1, 0.0);
#if 0
    float opts[5] = { 1e-3, 1e-12, 1e-12, 1e-12, 1e-6 };
    int iters = slevmar_dif(cost_func_expression<EnergyFunction>, &(params[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, engfun.get());
#else
    float opts[4] = { 1e-3, 1e-9, 1e-9, 1e-9 };
    int iters = slevmar_der(cost_func_expression<EnergyFunction>, jac_func_expression<EnergyFunction>, 
      &(wexp[0]), &(meas[0]), nparams, npts + 1, 128, NULL, NULL, NULL, NULL, engfun.get());
#endif

    //cout << "finished in " << iters << " iterations." << endl;

    float diff = 0;
    for (int i = 0; i < nparams; i++) {
      diff = max(diff, fabs(params.Wexp(i) - wexp[i]));
      params.Wexp(i) = wexp[i];
      //cout << wexp[i] << ' ';
    }
    //cout << endl;

    return diff < 1e-6;
  }

  void generateFittedMesh() {
    auto &tplt = model.tm;
    int nverts = tplt.length() / 3;
    arma::fmat pt(3, nverts);
    for (int i = 0, idx = 0; i < nverts; i++, idx += 3) {
      pt(0, i) = tplt(idx);
      pt(1, i) = tplt(idx + 1);
      pt(2, i) = tplt(idx + 2);
    }

    // batch rotation processing
    arma::fmat pt_trans = params.R * pt;
    #pragma omp parallel for
    for (int i = 0; i < nverts; i++) {
      int idx = i * 3;
      tmesh(idx) = pt_trans(0, i) + params.T(0);
      tmesh(idx + 1) = pt_trans(1, i) + params.T(1);
      tmesh(idx + 2) = pt_trans(2, i) + params.T(2);
    }
  }

private:
  MultilinearModel<float> &model;
  Parameters &params;
  unique_ptr<EnergyFunction> engfun;

  MultilinearModel<float> model_projected;
  vector<Constraint> cons;
  Tensor1<float> tmesh;
};
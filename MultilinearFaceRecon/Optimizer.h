#pragma once

#include "Math/Tensor.hpp"
#include "MultilinearModel.h"
#include "PointConstraint.h"
#include "EnergyFunctions.h"
#include "OffscreenRenderer.h"

enum FittingOption {
  FIT_POSE = 0,
  FIT_IDENTITY,
  FIT_EXPRESSION,
  FIT_POSE_AND_IDENTITY,
  FIT_POSE_AND_EXPRESSION,
  FIT_ALL,
  FIT_ALL_PROGRESSIVE
};

// wrappers
template <typename T, typename VT>
void cost_func_pose(VT *p, VT *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_pose(p, hx, m, n, NULL);
}

template <typename T, typename VT>
void jac_func_pose(VT *p, VT *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_pose(p, J, m, n, NULL);
}

template <typename T, typename VT>
void cost_func_identity(VT *p, VT *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_identity(p, hx, m, n, NULL);
}

template <typename T, typename VT>
void jac_func_identity(VT *p, VT *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_identity(p, J, m, n, NULL);
}

template <typename T, typename VT>
void cost_func_expression(VT *p, VT *hx, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->cost_expression(p, hx, m, n, NULL);
}

template <typename T, typename VT>
void jac_func_expression(VT *p, VT *J, int m, int n, void* adata) {
  T* engfun = static_cast<T*>(adata);
  engfun->jacobian_expression(p, J, m, n, NULL);
}

template <typename Model, typename Constraint, typename Parameters, typename EnergyFunction>
class Optimizer {
public:
  Optimizer(){}
  Optimizer(Model &model, Parameters &params) :model(model), params(params){
    // initialize the mesh tensor
    params.updateMesh(model);
    loadContourPoints();
  }
  ~Optimizer(){}

  void loadContourPoints(){
    cout << "loading contour points ..." << endl;
    const string filename = "C:/Users/Peihong/Desktop/Code/MultilinearFaceRecon/Data/contourpoints.txt";
    const int nContourRows = 75;
    contourPoints.resize(nContourRows);
    ifstream fin(filename);
    for (int i = 0; i < nContourRows; ++i) {
      string line;
      std::getline(fin, line);
      stringstream ss;
      ss << line;
      while (ss) {
        int x;
        ss >> x;
        contourPoints[i].push_back(x);
        contourVerts.push_back(x);
      }
      //cout << "#pts = " << contourPoints[i].size() << "\t" << line << endl;
    }
    cout << "done." << endl;
  }

  void bindOffscreenRenderer(const shared_ptr<OffscreenRenderer> &r) {
    renderer = r;
  }
  void init() {
    useInputBinding = true;
    updateProjectedTensors();
    initializeRigidTransformation();
    engfun.reset(new EnergyFunction(params));
  }

  Tensor1<double> fit(const vector<Constraint> &constraints,
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
    case FIT_ALL_PROGRESSIVE:
      //fit_all_progressive_expression_first();
      fit_all_progressive();
      break;
    }
    generateFittedMesh();

    return params.tmesh;
  }

private:
  void checkAndUpdateConstraints(const vector<Constraint> &constraints) {
    bool needInitialize = false;
    if (constraints.size() != params.constraintCount()) {
      // direct update
      needInitialize = true;
    }
    else{
#if 0
      // check every single constraint
      for (int i = 0; i < constraints.size(); ++i) {
        if (!cons[i].hasSameId(constraints[i])) {
          needInitialize = false;
          break;
        }
      }
#endif
    }
    if (needInitialize || useInputBinding) {
      params.setConstraints(constraints);
      useInputBinding = false;
    }
    else {
      // correspondences are updated, only update the locations of the constraints
      params.updateConstraints(constraints);
    }

    if (needInitialize) {
      init();
    }      
  }

  void updateProjectedTensors() {
    cout << "updating projected tensor ..." << endl;
#if 0
    vector<int> indices(cons.size());
    for (int i = 0; i < cons.size(); ++i) {
      indices[i] = cons[i].vidx;
    }
    model_projected = model.project(indices);
#else
    params.model_projected = model.project(params.getConstraintIndices());
#endif
    //cout << model_projected.core.dim(0) << ", " << model_projected.core.dim(1) << ", " << model_projected.core.dim(2) << endl;
    params.model_projected.applyWeights(params.Wid, params.Wexp);

    // assume no rotation at this point
    params.model_projected.tm0RT = params.model_projected.tm0;
    params.model_projected.tm1RT = params.model_projected.tm1;
  }

  void initializeRigidTransformation() {
    // estimate a position
    // no rotation initially
    params.RTparams[0] = 0; params.RTparams[1] = 0; params.RTparams[2] = 0;
    // at origin initially
    params.RTparams[3] = -0.05; params.RTparams[4] = -0.04; params.RTparams[5] = -0.95;

    params.R = arma::mat(3, 3);
    params.R(0, 0) = 1.0, params.R(1, 1) = 1.0, params.R(2, 2) = 1.0;
    params.T = arma::vec(3);

    params.Rmat = PhGUtils::Matrix3x3d::identity();
    params.Tvec = PhGUtils::Point3d::zero();
  }

  float computeError()
  {
    return engfun->error();
  }

  void updateContourPoints_ICP();

  void updateContourPoints();

  void fit_pose() {
    PhGUtils::message("fitting pose ...");
    params.fitExpression = false;

    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 16;
    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;

    while (!converged && iters++ < MAXITERS) {
      converged = true;

      converged &= fitRigidTransformation(32);
      
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      
      generateFittedMesh();
      /*
      renderer->updateMesh(params.tmesh);
      renderer->updateFocalLength(params.camparams.fy);
      renderer->render();
      */

      // update correspondence for contour feature points
      updateContourPoints();
      updateProjectedTensors();

      //QApplication::processEvents();
      //::system("pause");
    }
    cout << "Error = " << E << endl;
  }

  void fit_identity(){

  }
  
  void fit_expression() {

  }

  void fit_pose_and_identity() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;
    params.fitExpression = false;
    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 128;
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
      params.model_projected.transformTM1(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights();
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.model_projected.updateTMWithMode1(params.Wid);
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

      model.updateTMWithMode1(params.Wid);
      generateFittedMesh();
      renderer->updateMesh(params.tmesh);
      renderer->updateFocalLength(params.camparams.fy);
      renderer->render();

      updateContourPoints();
      updateProjectedTensors();
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    params.model_projected.updateTM0(params.Wid);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.Wid);
    model.updateTMWithMode1(params.Wid);
    timerOther.toc();
  }

  void fit_pose_and_expression() {
    params.fitExpression = true;

    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 128;
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
      params.model_projected.transformTM0(params.Rmat);
      timerOther.toc();

      timerExp.tic();
      converged &= fitExpressionWeights();
      timerExp.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.model_projected.updateTMWithMode0(params.Wexp);
      timerOther.toc();
      
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      // uncomment to show the transformation process		
      model.updateTMWithMode0(params.Wexp);
      generateFittedMesh();
      renderer->updateMesh(params.tmesh);
      renderer->updateFocalLength(params.camparams.fy);
      renderer->render();

      updateContourPoints();
      updateProjectedTensors();
    }

    timerTransform.tic();
    model.updateTMWithMode0(params.Wexp);
    timerTransform.toc();
  }

  void fit_all_progressive() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 1;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 16;

    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;
    int maxIters_id, maxIters_rt, maxIters_exp;
    double max_prior = 4.0, min_prior = 0.000001;

    while (!converged && iters++ <= MAXITERS) {

      params.w_prior_id = pow(max_prior + (min_prior - max_prior)*iters / (float)MAXITERS, 1.0/5.0);
      params.w_prior_exp = pow(max_prior + (min_prior - max_prior)*iters / (float)MAXITERS, 1.0/5.0);

      converged = true;

      maxIters_rt = 32;
      maxIters_id = min(iters * 2.5, 5.0 * pow(iters, 0.25));
      maxIters_exp = min(iters * 2.5, 5.0 * pow(iters, 0.25));

      timerRT.tic();
      converged &= fitRigidTransformation(maxIters_rt);
      timerRT.toc();

      // apply the new global transformation to tm1c
      // because tm1c is required in fitting identity weights
      timerOther.tic();
      params.model_projected.transformTM1(params.Rmat);
      timerOther.toc();

      timerID.tic();
      for (int fitidx = 0; fitidx < 20; ++fitidx) {
        converged &= fitIdentityWeights(0.05+fitidx*0.0025, maxIters_id);
      }
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.model_projected.updateTMWithMode1(params.Wid);
      params.model_projected.updateTM0(params.Wid);
      timerOther.toc();

      timerRT.tic();
      converged &= fitRigidTransformation(maxIters_rt);
      timerRT.toc();

      timerOther.tic();
      params.model_projected.transformTM0(params.Rmat);
      timerOther.toc();

      timerID.tic();
      for (int fitidx = 0; fitidx < 5; ++fitidx) {
        converged &= fitExpressionWeights(0.15 + fitidx * 0.025, maxIters_exp);
      }
      timerID.toc();

      model.updateTM1(params.Wexp);
      model.updateTMWithMode0(params.Wexp);

      /// optimize for camera parameters
      converged &= fitCameraParameters();

      timerOther.tic();
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      generateFittedMesh();
      renderer->updateMesh(params.tmesh);
      //renderer->updateFocalLength(params.camparams.fy);
      //renderer->render();

      updateContourPoints();
      updateProjectedTensors();
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    params.model_projected.updateTM0(params.Wid);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.Wid);
    model.updateTMWithMode1(params.Wid);
    timerOther.toc();
  }

  void fit_all_progressive_expression_first() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 128;
    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;
    int maxIters_id, maxIters_rt, maxIters_exp;

    while (!converged && iters++ < MAXITERS) {
      converged = true;

      maxIters_rt = 30;
      maxIters_id = min(iters * 1.05, 30);
      maxIters_exp = min(iters * 1.05, 30);

      timerRT.tic();
      converged &= fitRigidTransformation(maxIters_rt);
      timerRT.toc();

      // fit expression first
      timerOther.tic();
      model_projected.transformTM0(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitExpressionWeights(0.5, maxIters_exp);
      timerID.toc();

      model_projected.updateTM1(params.Wexp);
      model_projected.updateTMWithMode0(params.Wexp);

      // apply the new global transformation to tm1c
      // because tm1c is required in fitting identity weights
      timerOther.tic();
      model_projected.transformTM1(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights(0.5, maxIters_id);
      timerID.toc();

      model.updateTM0(params.Wid);
      model.updateTMWithMode1(params.Wid);

      /// optimize for camera parameters
      converged &= fitCameraParameters();

      timerOther.tic();
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      /*
      generateFittedMesh();
      renderer->updateMesh(tmesh);
      renderer->updateFocalLength(params.camparams.fy);
      renderer->render();

      updateContourPoints();
      updateProjectedTensors();
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

  void fit_all() {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 0;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = 128;
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
      params.model_projected.transformTM1(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights();
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.model_projected.updateTMWithMode1(params.Wid);
      params.model_projected.updateTM0(params.Wid);
      params.model_projected.transformTM0(params.Rmat);
      timerOther.toc();

      timerID.tic();
      converged &= fitExpressionWeights();
      timerID.toc();

      model.updateTM1(params.Wexp);
      model.updateTMWithMode0(params.Wexp);

      /// optimize for camera parameters
      converged &= fitCameraParameters();

      timerOther.tic();
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      /*
      generateFittedMesh();
      renderer->updateMesh(tmesh);
      renderer->updateFocalLength(params.camparams.fy);
      renderer->render();

      updateContourPoints();
      updateProjectedTensors();
      */
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    params.model_projected.updateTM0(params.Wid);
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.Wid);
    model.updateTMWithMode1(params.Wid);
    timerOther.toc();
  }

  bool fitCameraParameters() {
    params.updateCameraParameters(params.model_projected);
    return true;
  }

  bool fitRigidTransformation(int maxiters=128) {
    //cout << "fitting rigid transformation ..." << endl;
    int npts = params.constraintCount();
    vector<double> meas(npts, 0.0);
    auto &RTparams = params.RTparams;
    int nparams = Parameters::nRTparams;

    double opts[4] = { 1.0, 1e-12, 1e-12, 1e-12 };  // tau, eps1, eps2, eps3
#if 0
    int iters = dlevmar_der(cost_func_pose<EnergyFunction>, jac_func_pose<EnergyFunction>, 
      RTparams, &(meas[0]), nparams, npts, maxiters, opts, NULL, NULL, NULL, engfun.get());
#else
    int iters = dlevmar_dif(cost_func_pose<EnergyFunction>, RTparams, &(meas[0]), nparams, npts, 256, NULL, NULL, NULL, NULL, engfun.get());
#endif

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

  bool fitIdentityWeights(float underrelax_factor = 1.0, int maxiters = 128) {
    params.fitExpression = false;
    int nparams = params.model_projected.core.dim(0);
    vector<double> wid(params.Wid.rawptr(), params.Wid.rawptr() + nparams);
    int npts = params.constraintCount();
    vector<double> meas(npts + 1, 0.0);
#if 0
    double opts[5] = { 1e-3, 1e-9, 1e-9, 1e-9, 1e-6 };
    int iters = dlevmar_dif(cost_func_identity<EnergyFunction>, &(wid[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, engfun.get());
#else
    // evaluate the jacobian
    /*
    vector<double> errorvec(npts + 1);
    dlevmar_chkjac(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), nparams, npts + 1, engfun.get(), &(errorvec[0]));
    PhGUtils::message("error vec[identity]");
    PhGUtils::printArray(&(errorvec[0]), npts + 1);
    */
    double opts[4] = { 10.0, 1e-9, 1e-9, 1e-9 };
    int iters = dlevmar_der(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), &(meas[0]), nparams, npts + 1, maxiters, opts, NULL, NULL, NULL, engfun.get());
#endif
    //cout << "identity fitting finished in " << iters << " iterations." << endl;

    //debug("rtn", rtn);
    double diff = 0.0;
    //b.print("b");
    for (int i = 0; i < nparams; i++) {
      diff = max(diff, fabs(params.Wid(i) - wid[i]) / (fabs(params.Wid(i)) + 1e-12f));
      params.Wid(i) = underrelax_factor * wid[i] + (1.0 - underrelax_factor) * params.Wid(i);
      //cout << params.Wid(i) << ' ';
    }
    //cout << endl;

    //cout << "diff = " << diff << endl;

    return diff < 1e-4;
  }

  bool fitExpressionWeights(float underrelax_factor=1.0, int maxiters=128) {
    params.fitExpression = true;
    // fix both rotation and identity weights, solve for expression weights
    int nparams = params.model_projected.core.dim(1);
    vector<double> wexp(params.Wexp.rawptr(), params.Wexp.rawptr() + nparams);
    int npts = params.constraintCount();
    vector<double> meas(npts + 1, 0.0);
#if 0
    double opts[5] = { 1e-3, 1e-9, 1e-9, 1e-9, 1e-6 };
    int iters = dlevmar_dif(cost_func_expression<EnergyFunction>, &(wexp[0]), &(meas[0]), nparams, npts + 1, 128, opts, NULL, NULL, NULL, engfun.get());
#else
    double opts[4] = { 1.0, 1e-9, 1e-9, 1e-9 };
    int iters = dlevmar_der(cost_func_expression<EnergyFunction>, jac_func_expression<EnergyFunction>, 
      &(wexp[0]), &(meas[0]), nparams, npts + 1, maxiters, NULL, NULL, NULL, NULL, engfun.get());
#endif

    //cout << "finished in " << iters << " iterations." << endl;

    double diff = 0;
    for (int i = 0; i < nparams; i++) {
      diff = max(diff, fabs(params.Wexp(i) - wexp[i])/(fabs(params.Wexp(i))+1e-12));
      params.Wexp(i) = underrelax_factor * wexp[i] + (1.0-underrelax_factor)*params.Wexp(i);
      //cout << wexp[i] << ' ';
    }
    //cout << endl;

    return diff < 1e-4;
  }

  void generateFittedMesh() {
    params.generateFittedMesh(model);
  }

public:
  const vector<Constraint>& getConstraints() const { return params.cons; }

private:
  Model &model;
  Parameters &params;
  unique_ptr<EnergyFunction> engfun;

  shared_ptr<OffscreenRenderer> renderer;
  bool useInputBinding;
  vector<vector<int>> contourPoints;
  vector<int> contourVerts;
};

template <typename Model, typename Constraint, typename Parameters, typename EnergyFunction>
void Optimizer<Model, Constraint, Parameters, EnergyFunction>::updateContourPoints()
{
  cout << "updating contour points..." << endl;
#if 1
  params.updateContourPoints(contourPoints, contourVerts);
#else
  PhGUtils::QuadMesh &baseMesh = renderer->getBaseMesh();
  baseMesh.computeNormals(contourVerts);

  const PhGUtils::Vector3f viewVec(0, 0, -1);
  int imgWidth = renderer->getWidth(), imgHeight = renderer->getHeight();

  // for each row, pick the best candidates
  int nContourRows = contourPoints.size();
  vector<int> candidates;
  for (int i = 0; i < nContourRows; ++i) {
    int minIdx = 0;
    float minProd = fabs(baseMesh.normal(contourPoints[i].front()).dot(viewVec));
    for (int j = 1; j < contourPoints[i].size(); ++j) {
      auto &nij = baseMesh.normal(contourPoints[i][j]);
      float prod = fabs(nij.dot(viewVec));
      if (prod < minProd) {
        minIdx = j;
        minProd = prod;
      }
    }

    candidates.push_back(contourPoints[i][minIdx]);
    if (minIdx < contourPoints[i].size()-1)
      candidates.push_back(contourPoints[i][minIdx+1]);
    if (minIdx > 0)
      candidates.push_back(contourPoints[i][minIdx-1]);
  }

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
      renderer->project(uf, vf, df, fpt.x, fpt.y, fpt.z);
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
#endif
}

#if 0
template <typename Model, typename Constraint, typename Parameters, typename EnergyFunction>
void Optimizer<Model, Constraint, Parameters, EnergyFunction>::updateContourPoints_ICP()
{
  //PhGUtils::message("Updating contour points...");
  vector<Constraint> contour;
  int imgWidth = renderer->getWidth(), imgHeight = renderer->getHeight();

  auto depthMap = renderer->getDepthMap();
  auto indexMap = renderer->getIndexMap();
  auto baseMesh = renderer->getBaseMesh();

  contour.reserve(32);
  // project the contour points to the image plane
  vector<PhGUtils::Point3f> contourPts;
  int contourPointsCount = 15;
  for (int i = 0; i < contourPointsCount; ++i) {
    int fptIdx = cons[i].vidx;
    int fptOffset = fptIdx * 3;
    PhGUtils::Point3f fpt(tmesh(fptOffset), tmesh(fptOffset + 1), tmesh(fptOffset + 2));
    double u, v, d;
    renderer->project(u, v, d, fpt.x, fpt.y, fpt.z);
    contourPts.push_back(PhGUtils::Point3f(u, v, d));
  }

  // on the projected image plane, find the closest match
  for (int i = 0; i < contourPts.size(); ++i) {
    // do not change this point
    if (i == 7) continue;

    const int wSize = 5;

    set<int> checkedFaces;
    set<int> fpts_candidate;

    const float alpha = 0.75;
    int u = alpha * cons[i].q.x + (1.0 - alpha)*contourPts[i].x;
    //cout << cons[i].q.x << " vs " << contourPts[i].x << "\t";
    //cout << cons[i].q.y << " vs " << (imgHeight-1-contourPts[i].y) << endl;
    int v = alpha * cons[i].q.y + (1.0 - alpha)*(imgHeight-1-contourPts[i].y);

    // scan a small window to find the potential faces
    for (int r = v - wSize; r <= v + wSize; r++) {
      int rr = imgHeight - 1 - r;
      for (int c = u - wSize; c <= u + wSize; c++) {
        int pidx = rr * imgWidth + c;
        int poffset = pidx << 2;		// pixel index for synthesized image

        // get face index and depth value
        int fidx;
        PhGUtils::decodeIndex<float>(indexMap[poffset] / 255.0f, indexMap[poffset + 1] / 255.0f, indexMap[poffset + 2] / 255.0f, fidx);
        float depthVal = depthMap[pidx];

        if (depthVal < 1.0) {
          if (std::find(checkedFaces.begin(), checkedFaces.end(), fidx) == checkedFaces.end()) {
            // not checked yet
            checkedFaces.insert(fidx);
            const PhGUtils::QuadMesh::face_t& f = baseMesh.face(fidx);

            fpts_candidate.insert(f.x); fpts_candidate.insert(f.y); fpts_candidate.insert(f.z); fpts_candidate.insert(f.w);
          }
          else {
            // already checked, do nothing
          }
        }
      }

      //cout << "candidates = " << fpts_candidate.size() << endl;
    }

    // now check the candidates
    float dx = contourPts[i].x - u;
    float dy = (imgHeight - 1 - contourPts[i].y) - v;
    float closestDist = dx*dx + dy*dy;
    int closestIdx = cons[i].vidx;

    for (auto cidx : fpts_candidate) {
      int cOffset = cidx * 3;
      PhGUtils::Point3f fpt(tmesh(cOffset), tmesh(cOffset + 1), tmesh(cOffset + 2));
      double uf, vf, df;
      renderer->project(uf, vf, df, fpt.x, fpt.y, fpt.z);
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
    //cout << fpts_candidate.size() << "\t" << cons[i].vidx << " -> " << closestIdx << endl;
    cons[i].vidx = closestIdx;
  }
}
#endif
#pragma once

#include "Math/Tensor.hpp"
#include "MultilinearModel.h"
#include "PointConstraint.h"
#include "EnergyFunctions.h"
#include "OffscreenRenderer.h"

enum FittingType {
  FIT_POSE = 0,
  FIT_IDENTITY,
  FIT_EXPRESSION,
  FIT_POSE_AND_IDENTITY,
  FIT_POSE_AND_EXPRESSION,
  FIT_ALL,
  FIT_ALL_PROGRESSIVE
};

struct FittingOptions {
  FittingOptions(FittingType type) :type(type), maxIters(16){}
  FittingOptions(FittingType type, int maxIters) :type(type), maxIters(maxIters){}

  FittingType type;
  int maxIters;
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

  vector<Tensor1<double>> fit(const vector<vector<Constraint>> &constraints, FittingOptions &ops) {
    checkAndUpdateConstraints(constraints);
    switch (ops.type) {
    case FIT_POSE:
      fit_pose();
      break;
    case FIT_ALL_PROGRESSIVE:
      fit_all_progressive(ops.maxIters);
      break;
    default:
      break;
    }
    generateFittedMesh();

    auto focalLengths = params.getFocalLength();
    for (auto x : focalLengths) {
      cout << x << endl;
    }
    return params.getFittedMeshes();
  }

  Tensor1<double> fit(const vector<Constraint> &constraints,
                     FittingOptions &ops) {

    // check if we need to update the projected tensor
    checkAndUpdateConstraints(constraints);

    switch (ops.type) {
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
      fit_all_progressive(ops.maxIters);
      break;
    }
    generateFittedMesh();

    return params.reconParams.tmesh;
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

  void checkAndUpdateConstraints(const vector<vector<Constraint>> &constraints) {
    bool needInitialize = !params.isConstraintsSet();
    cout << "need initialization = " << needInitialize << endl;
    if (needInitialize || useInputBinding) {
      params.setConstraints(constraints);
      useInputBinding = false;
    }
    else {
      // correspondences are updated, only update the locations of the constraints
      cout << "updating constraints ..." << endl;
      params.updateConstraints(constraints);
    }
    if (needInitialize) {
      cout << "initializing constraints ..." << endl;
      init();
    }
    system("pause");
  }

  void updateProjectedTensors() {
    //cout << "updating projected tensor ..." << endl;
    params.updateProjectedTensor(model);
    //cout << "done." << endl;
  }

  void initializeRigidTransformation() {
    params.initializeRigidTransformation();
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
    const int MAXITERS = 2;
    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;

    while (!converged && iters++ < MAXITERS) {
      converged = true;

      converged &= fitRigidTransformation(256/(float)(iters+1), 5e3);
      
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      
      params.generateFittedMesh(model, false);

      // update correspondence for contour feature points
      updateContourPoints();
      updateProjectedTensors();
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
      params.transformTM1();
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights();
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.updateTMWithMode1();
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

      updateContourPoints();
      updateProjectedTensors();
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    params.updateTM0();
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.modelParams.Wid);
    model.updateTMWithMode1(params.modelParams.Wid);
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
      params.transformTM0();
      timerOther.toc();

      timerExp.tic();
      converged &= fitExpressionWeights();
      timerExp.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.updateTMWithMode0();
      timerOther.toc();
      
      E = computeError();
      PhGUtils::info("iters", iters, "Error", E);

      converged |= E < errorThreshold;
      converged |= fabs(E - E0) < errorDiffThreshold;
      E0 = E;
      timerOther.toc();

      updateContourPoints();
      updateProjectedTensors();
    }

    timerTransform.tic();
    model.updateTMWithMode0(params.modelParams.Wexp);
    timerTransform.toc();
  }

  void fit_all_progressive(int maxIters = 16) {
    PhGUtils::Timer timerRT, timerID, timerExp, timerOther, timerTransform, timerTotal;

    timerTotal.tic();
    int iters = 1;
    float E0 = 0, E = 1e10;
    bool converged = false;
    const int MAXITERS = maxIters;

    const float errorThreshold = 1e-3;
    const float errorDiffThreshold = 1e-6;
    int maxIters_id, maxIters_rt, maxIters_exp;
    double max_prior = 4.0, min_prior = 0.0025;

    while (!converged && iters++ <= MAXITERS) {

      params.priorParams.w_prior_id = pow(max_prior + (min_prior - max_prior)*iters / (float)MAXITERS, 1.0 / 3.0);
      params.priorParams.w_prior_exp = pow(max_prior + (min_prior - max_prior)*iters / (float)MAXITERS, 1.0 / 4.0);

      converged = true;

      maxIters_rt = min((128-iters) * 2.5, 5.0 * pow((128-iters), 0.25));
      maxIters_id = min(iters * 2.5, 5.0 * pow(iters, 0.25));
      maxIters_exp = min(iters * 2.5, 5.0 * pow(iters, 0.25));
      int niters_rt = 5 + iters * 2.5;
      int niters_id = 5 + iters * 2.5;
      int niters_exp = 5 + iters * 2.5;

      timerRT.tic();
      bool rtCoverged = false;
      for (int i = 0; i < niters_rt; ++i) {
        rtCoverged |= fitRigidTransformation(maxIters_rt, 1e4);
        if (rtCoverged) break;
      }
      converged &= rtCoverged;
      timerRT.toc();

      // apply the new global transformation to tm1c
      // because tm1c is required in fitting identity weights
      timerOther.tic();
      params.transformTM1();
      timerOther.toc();

      timerID.tic();
      for (int fitidx = 0; fitidx < niters_id; ++fitidx) {
        converged &= fitIdentityWeights(0.15, maxIters_id);
      }
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.updateTMWithMode1();
      params.updateTM0();
      timerOther.toc();

      timerRT.tic();
      rtCoverged = false;
      for (int i = 0; i < niters_rt; ++i) {
        rtCoverged |= fitRigidTransformation(maxIters_rt);
        if (rtCoverged) break;
      }
      converged &= rtCoverged;
      timerRT.toc();

      timerOther.tic();
      params.transformTM0();
      timerOther.toc();

      timerID.tic();
      for (int fitidx = 0; fitidx < niters_exp; ++fitidx) {
        converged &= fitExpressionWeights(0.075, maxIters_exp);
      }
      timerID.toc();

      timerOther.tic();
      params.updateTMWithMode0();
      params.updateTM1();
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

      // this is necessary for generating fitted mesh
      params.generateFittedMesh(model, true);
      
#if 1
      updateContourPoints();
      updateProjectedTensors();
#endif
    }

    timerOther.tic();
    // update tm0c with the new identity weights
    // now the tensor is not updated with global rigid transformation
    params.updateTM0();    
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();

    // update TM0
    //model.updateTM0(params.modelParams.Wid);
    //model.updateTMWithMode1(params.modelParams.Wid);
    params.generateFittedMesh(model, true);
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
      params.transformTM1();
      timerOther.toc();

      timerID.tic();
      converged &= fitIdentityWeights();
      timerID.toc();

      // compute tmc from the new tm1c or new tm0c
      timerOther.tic();
      params.updateTMWithMode1();
      params.updateTM0();
      params.transformTM0();
      timerOther.toc();

      timerID.tic();
      converged &= fitExpressionWeights();
      timerID.toc();

      model.updateTM1(params.modelParams.Wexp);
      model.updateTMWithMode0(params.modelParams.Wexp);

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
    params.updateTM0();
    //corec.modeProduct(Wid, 0, tm0c);
    timerOther.toc();

    timerOther.tic();
    // update TM0
    model.updateTM0(params.modelParams.Wid);
    model.updateTMWithMode1(params.modelParams.Wid);
    timerOther.toc();
  }

  bool fitCameraParameters() {
    params.updateCameraParameters();
    return true;
  }

  bool fitRigidTransformation(int maxiters=128, double mu=1e3) {
    //cout << "fitting rigid transformation ..." << endl;
    int nmeas = params.constraintCount();
    vector<double> meas(nmeas, 0.0);
    auto RTparams = params.getRTParams();
    int nparams = params.getNDims_rt();
#if 1
    /*
    vector<double> errorvec(nmeas);
    dlevmar_chkjac(cost_func_pose<EnergyFunction>, jac_func_pose<EnergyFunction>, RTparams, nparams, nmeas, engfun.get(), &(errorvec[0]));
    PhGUtils::message("error vec[rt]");
    PhGUtils::printArray(&(errorvec[0]), nmeas);
    */

    double opts[4] = { mu, 1e-12, 1e-12, 1e-12 };  // tau, eps1, eps2, eps3
    int iters = dlevmar_der(cost_func_pose<EnergyFunction>, jac_func_pose<EnergyFunction>, 
      RTparams, &(meas[0]), nparams, nmeas, maxiters, opts, NULL, NULL, NULL, engfun.get());
#else
    double opts[5] = { mu, 1e-12, 1e-12, 1e-12, 1e-6 };  // tau, eps1, eps2, eps3
    int iters = dlevmar_dif(cost_func_pose<EnergyFunction>, RTparams, &(meas[0]), nparams, nmeas, maxiters, opts, NULL, NULL, NULL, engfun.get());
#endif

    //cout << "iters = " << iters << endl;
    // set up the matrix and translation vector
    double diff = params.updateRTParams();

    /*
    cout << "after" << endl;
    cout << params.R << endl;
    cout << params.T << endl;
    */
    return diff < 1e-6;
  }

  bool fitIdentityWeights(float underrelax_factor = 1.0, int maxiters = 128) {
    params.fitExpression = false;
    int nparams = params.getNDims_id();
    auto pWid = params.getWidParams();
    vector<double> wid(pWid, pWid + nparams);
    int nmeas = params.constraintCountWithPrior();
    vector<double> meas(nmeas, 0.0);
#if 0
    double opts[5] = { 100.0, 1e-9, 1e-9, 1e-9, 1e-6 };
    int iters = dlevmar_dif(cost_func_identity<EnergyFunction>, &(wid[0]), &(meas[0]), nparams, nmeas, maxiters, opts, NULL, NULL, NULL, engfun.get());
#else
    // evaluate the jacobian
    /*
    vector<double> errorvec(nmeas);
    dlevmar_chkjac(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), nparams, nmeas, engfun.get(), &(errorvec[0]));
    PhGUtils::message("error vec[identity]");
    PhGUtils::printArray(&(errorvec[0]), nmeas);
    */
    double opts[4] = { 100.0, 1e-9, 1e-9, 1e-9 };
    int iters = dlevmar_der(cost_func_identity<EnergyFunction>, jac_func_identity<EnergyFunction>, &(wid[0]), &(meas[0]), nparams, nmeas, maxiters, opts, NULL, NULL, NULL, engfun.get());
#endif
    //cout << "identity fitting finished in " << iters << " iterations." << endl;

    //debug("rtn", rtn);
    double diff = params.updateWidParams(wid, underrelax_factor);
    //cout << endl;

    //cout << "diff = " << diff << endl;

    return diff < 1e-6;
  }

  bool fitExpressionWeights(float underrelax_factor=1.0, int maxiters=128) {
    params.fitExpression = true;
    // fix both rotation and identity weights, solve for expression weights
    int nparams = params.getNDims_exp();
    auto pWexp = params.getWexpParams();
    vector<double> wexp(pWexp, pWexp + nparams);
    int nmeas = params.constraintCountWithPrior();
    vector<double> meas(nmeas, 0.0);
#if 0
    double opts[5] = { 100.0, 1e-9, 1e-9, 1e-9, 1e-6 };
    int iters = dlevmar_dif(cost_func_expression<EnergyFunction>, &(wexp[0]), &(meas[0]), nparams, nmeas, maxiters, opts, NULL, NULL, NULL, engfun.get());
#else
    /*
    vector<double> errorvec(nmeas);
    dlevmar_chkjac(cost_func_expression<EnergyFunction>, jac_func_expression<EnergyFunction>, &(wexp[0]), nparams, nmeas, engfun.get(), &(errorvec[0]));
    PhGUtils::message("error vec[expression]");
    PhGUtils::printArray(&(errorvec[0]), nmeas);
    */
    double opts[4] = { 100.0, 1e-9, 1e-9, 1e-9 };
    int iters = dlevmar_der(cost_func_expression<EnergyFunction>, jac_func_expression<EnergyFunction>, 
      &(wexp[0]), &(meas[0]), nparams, nmeas, maxiters, NULL, NULL, NULL, NULL, engfun.get());
#endif

    //cout << "finished in " << iters << " iterations." << endl;
    double diff = params.updateWexpParams(wexp, underrelax_factor);
    //cout << endl;

    return diff < 1e-6;
  }

  void generateFittedMesh() {
    params.generateFittedMesh(model);
  }

public:
  const vector<Constraint>& getConstraints() const { return params.reconParams.cons; }

private:
  Model &model;   // the core tensor
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
  //cout << "updating contour points..." << endl;
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
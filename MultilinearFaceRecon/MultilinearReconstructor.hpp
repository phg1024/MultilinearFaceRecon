#pragma once

#include "Math/Tensor.hpp"
#include "Geometry/point.hpp"
#include "Geometry/matrix.hpp"
#include "Geometry/geometryutils.hpp"
#include "Geometry/Mesh.h"
#include "Geometry/MeshLoader.h"

// levmar header
#include "levmar.h"

#include <cula.h>

#include "Math/DenseVector.hpp"
#include "Math/DenseMatrix.hpp"
#include "Math/denseblas.h"

#include "Utils/utility.hpp"

#include <QGLWidget>
#include <QGLFramebufferObject>

#include <concurrent_vector.h>

#include "PointConstraint.h"
#include "Optimizer.h"
#include "MultilinearModel.h"
#include "Parameters.h"
#include "OffscreenRenderer.h"

template <class Model, class Constraint, class Optimizer, class Parameters = SingleImageParameters>
class MultilinearReconstructor {
public:
  MultilinearReconstructor() {
    mkl_set_num_threads(8);
    init();
  }
  ~MultilinearReconstructor(){}

  void reset(int n = 1);
  void setImageSize(int w, int h);
  void setImageSize(const vector<PhGUtils::Point2i> &sizes);
  void fit(const vector<Constraint> &constraints, FittingOptions ops) {
    tmesh = opt->fit(constraints, ops);
  }
    
  vector<Tensor1<double>> fit(const vector<vector<Constraint>> &constraints, FittingOptions ops)
  {
    return opt->fit(constraints, ops);
  }

  const Tensor1<double> &currentMesh() const{
    return tmesh;
  }

  PhGUtils::QuadMesh& getMesh(int idx) {
    return params.getMesh(idx);
  }

  const float getFocalLength(int idx) const {
    return params.getFocalLength(idx);
  }

  const void getTranslation(int idx, float &x, float &y, float &z) {
    params.getTranslation(idx, x, y, z);
  }

  const vector<Constraint>& getConstraints() const { return opt->getConstraints(); }

private:
  friend typename Optimizer;
  friend class BlendShapeViewer;
  void init() {
    // load the default mesh
    PhGUtils::OBJLoader loader;
    loader.load("../Data/shape_0.obj");
    baseMesh.initWithLoader(loader);

    params.loadPrior("../Data/blendshape/wid_double.bin", "../Data/blendshape/wexp_double.bin");
    params.init(0);
    params.setBaseMesh("../Data/shape_0.obj");

    loadModel("../Data/blendshape/core_double.bin");
    preprocessModel();
    opt.reset(new Optimizer(model, params));

    renderer.reset(new OffscreenRenderer(640, 480));
    opt->bindOffscreenRenderer(renderer);
  }

  void loadModel(const string &filename) {
    model = Model(filename);    
  }

  void preprocessModel() {
    model.unfold();
    params.updateMesh(model);
    model.applyWeights(Tensor1<double>::fromVec(params.priorParams.mu_wid0), Tensor1<double>::fromVec(params.priorParams.mu_wexp0));
  }

private:
  Model model;
  shared_ptr<Optimizer> opt;
  Tensor1<double> tmesh;   // fitted mesh
  Parameters params;
  shared_ptr<OffscreenRenderer> renderer;

  PhGUtils::QuadMesh baseMesh;
};

template <class Model, class Constraint, class Optimizer, class Parameters /*= SingleImageParameters*/>
void MultilinearReconstructor<Model, Constraint, Optimizer, Parameters>::setImageSize(const vector<PhGUtils::Point2i> &sizes)
{
  params.setImageSize(sizes);
}

template <class Model, class Constraint, class Optimizer, class Parameters /*= DefaultParameters*/>
void MultilinearReconstructor<Model, Constraint, Optimizer, Parameters>::setImageSize(int w, int h)
{
  // update the parameters for the reconstruction engine
  params.setImageSize(w, h);

  // update the parameters for the fbo
  renderer->resizeBuffers(w, h);
}

template <class Model, class Constraint, class Optimizer, class Parameters /*= DefaultParameters*/>
void MultilinearReconstructor<Model, Constraint, Optimizer, Parameters>::reset(int n)
{
  params.init(n);  
  params.setBaseMesh("../Data/shape_0.obj");
  opt->init();
}

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

template <class Constraint, class Optimizer, class Parameters = DefaultParameters>
class MultilinearReconstructor {
public:
  MultilinearReconstructor() {
    mkl_set_num_threads(8);
    init();
  }
  ~MultilinearReconstructor(){}

  void reset();
  void fit(const vector<Constraint> &constraints, FittingOption ops = FIT_ALL) {
    tmesh = opt->fit(constraints, ops);
  }

  const Tensor1<float> &currentMesh() const{
    return tmesh;
  }

private:
  friend typename Optimizer;
  friend class BlendShapeViewer;
  void init() {
    params.loadPrior("../Data/blendshape/wid.bin", "../Data/blendshape/wexp.bin");
    params.init();
    loadModel("../Data/blendshape/core.bin");
    preprocessModel();
    opt.reset(new Optimizer(model, params));
  }

  void loadModel(const string &filename) {
    model = MultilinearModel<float>(filename);    
  }

  void preprocessModel() {
    model.unfold();
    model.applyWeights(params.Wid, params.Wexp);
  }

private:
  MultilinearModel<float> model;
  shared_ptr<Optimizer> opt;
  Tensor1<float> tmesh;   // fitted mesh
  Parameters params;
};

template <class Constraint, class Optimizer, class Parameters /*= DefaultParameters*/>
void MultilinearReconstructor<Constraint, Optimizer, Parameters>::reset()
{
  this->params.init();
  model.applyWeights(params.Wid, params.Wexp);
  opt->init();
}

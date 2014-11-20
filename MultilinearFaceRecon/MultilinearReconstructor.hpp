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

template <class Constraint, class Optimizer, class Parameters = DefaultParameters>
class MultilinearReconstructor {
public:
  MultilinearReconstructor() {
    mkl_set_num_threads(8);
    init();
  }
  ~MultilinearReconstructor(){}

  void reset();
  void setImageSize(int w, int h);
  void fit(const vector<Constraint> &constraints, FittingOption ops = FIT_ALL) {
    tmesh = opt->fit(constraints, ops);
  }

  const Tensor1<double> &currentMesh() const{
    return tmesh;
  }

  const vector<Constraint>& getConstraints() const { return opt->getConstraints(); }

private:
  friend typename Optimizer;
  friend class BlendShapeViewer;
  void init() {
    params.loadPrior("../Data/blendshape/wid_double.bin", "../Data/blendshape/wexp_double.bin");
    params.init();
    loadModel("../Data/blendshape/core_double.bin");
    preprocessModel();
    opt.reset(new Optimizer(model, params));

    renderer.reset(new OffscreenRenderer(640, 480));
    opt->bindOffscreenRenderer(renderer);
  }

  void loadModel(const string &filename) {
    model = MultilinearModel<double>(filename);    
  }

  void preprocessModel() {
    model.unfold();
    model.applyWeights(params.Wid, params.Wexp);
  }

private:
  MultilinearModel<double> model;
  shared_ptr<Optimizer> opt;
  Tensor1<double> tmesh;   // fitted mesh
  Parameters params;
  shared_ptr<OffscreenRenderer> renderer;
};

template <class Constraint, class Optimizer, class Parameters /*= DefaultParameters*/>
void MultilinearReconstructor<Constraint, Optimizer, Parameters>::setImageSize(int w, int h)
{
  // update the parameters for the reconstruction engine
  params.camparams.cx = w / 2.0;
  params.camparams.cy = h / 2.0;
  params.camparams.fx = -1000.0;
  params.camparams.fy = 1000.0;

  // update the parameters for the fbo
  renderer->resizeBuffers(w, h);
}

template <class Constraint, class Optimizer, class Parameters /*= DefaultParameters*/>
void MultilinearReconstructor<Constraint, Optimizer, Parameters>::reset()
{
  this->params.init();
  model.applyWeights(params.Wid, params.Wexp);
  opt->init();
}

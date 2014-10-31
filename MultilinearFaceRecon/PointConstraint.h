#pragma once

#include "Geometry/geometryutils.hpp"

struct PointConstraint {
  int vidx; // vertex index on the mesh
  float weight;

  bool hasSameId(const PointConstraint &other) {
    return vidx == other.vidx;
  }
};

struct Constraint_2D : PointConstraint {
  static PhGUtils::Matrix3x3f MProj;

  Constraint_2D(){}
  ~Constraint_2D(){}

  PhGUtils::Point2f q;  // 2D point

  float residue(const PhGUtils::Point3f &p) {
    float u, v;
    PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Constraint_2D::MProj);
    float dx = q.x - u, dy = q.y - v;
    return sqrt(dx*dx + dy*dy) * weight;
  }
};

struct Constraint_2DAndDepth : public PointConstraint{
  static PhGUtils::Matrix3x3f Mproj;

  Constraint_2DAndDepth(){}
  ~Constraint_2DAndDepth(){}

  PhGUtils::Point3f q;  // 2D point plus depth
  float w_depth;

  float residue(const PhGUtils::Point3f &p) {
    float u, v, d;
    PhGUtils::projectPoint(u, v, d, p.x, p.y, p.z, Constraint_2DAndDepth::Mproj);
    float dx = q.x - u, dy = q.y - v, dz = q.z - d;
    return (sqrt(dx*dx + dy*dy) + fabs(dz) * w_depth) * weight;
  }
};

struct Constraint_3D : public PointConstraint{
  Constraint_3D(){}
  ~Constraint_3D(){}

  PhGUtils::Point3f q;  // 3D point

  float residue(const PhGUtils::Point3f &p) {
    return p.distanceTo(q) * weight;
  }
};
#pragma once

#include "MultilinearModel.h"
#include "Geometry/geometryutils.hpp"
#include "Parameters.h"

struct EnergyFunction2D {
  EnergyFunction2D():model(MultilinearModel<float>()), params(DefaultParameters()), targets_2d(vector<Constraint_2D>()){}
  EnergyFunction2D(MultilinearModel<float> &model, DefaultParameters &params, vector<Constraint_2D> &cons) :
    model(model), params(params), targets_2d(cons){
    f_pose = &EnergyFunction2D::cost_pose;
    jac_pose = &EnergyFunction2D::jacobian_pose;

    f_id = &EnergyFunction2D::cost_identity;
    jac_id = &EnergyFunction2D::jacobian_identity;

    f_exp = &EnergyFunction2D::cost_expression;
    jac_exp = &EnergyFunction2D::jacobian_expression;
  }

  typedef void (EnergyFunction2D::*fptr)(float *p, float *hx, int m, int n, void* adata);

  float error() {
    int npts = targets_2d.size();
    float E = 0;
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);
    auto &tmc = model.tm;
    for (int i = 0; i < npts; i++) {
      int vidx = i * 3;
      // should change this to a fast version
      PhGUtils::Point3f p(tmc(vidx), tmc(vidx + 1), tmc(vidx + 2));
      //cout << p << endl;
      /*p = Rmat * p + Tvec;*/
      PhGUtils::transformPoint(p.x, p.y, p.z, params.Rmat, params.Tvec);

      // project the point
      float u, v;
      PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Mproj);

      float dx, dy;
      dx = u - targets_2d[i].q.x;
      dy = v - targets_2d[i].q.y;
      E += (dx * dx + dy * dy);
    }

    E /= npts;
    return E;
  }

  void cost_pose(float *p, float *hx, int m, int n, void* adata) {
    float s, rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto &tmc = model.tm;

    // set up rotation matrix and translation vector
    PhGUtils::Point3f T(tx, ty, tz);
    PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);
    
    //cout << Mproj << endl;

    // apply the new global transformation
    for (int i = 0, vidx = 0; i < npts; i++) {
      //PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
      float wpt = 1.0;

      // exclude mouth region and chin region
      if (!params.fitExpression) {
        if (i >= 48 || (i >= 4 && i <= 12))
          wpt = 0.9;
      }
      else if (i < 17) wpt = 1.0;

      float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      const PhGUtils::Point2f& q = targets_2d.at(i).q;

      // PhGUtils::Point3f pp = R * p + T;
      PhGUtils::transformPoint(px, py, pz, R, T);
      //cout << px << " " << py << " " << pz << endl;

      // Projection to image plane
      float u, v;
      PhGUtils::projectPoint(u, v, px, py, pz, Mproj);

      //cout << i << "\t" << u << ", " << v << endl;

      float dx = q.x - u, dy = q.y - v;
      //cout << i << "\t" << dx << ", " << dy << ", " << dz << endl;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx * dx + dy * dy) * wpt;
    }
  }

  void jacobian_pose(float *p, float *J, int m, int n, void* adata) {
    // J is a n-by-m matrix
    float rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto &tmc = model.tm;

    // set up rotation matrix and translation vector
    PhGUtils::Matrix3x3f R = PhGUtils::rotationMatrix(rx, ry, rz);
    PhGUtils::Matrix3x3f Jx, Jy, Jz;
    PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);

    //cout << Jx << endl;
    //cout << Jy << endl;
    //cout << Jz << endl;

    // for Jacobian of projection/viewport transformation
    const float f_x = params.camparams.fx, f_y = params.camparams.fy;
    float Jf[6] = { 0 };

    // apply the new global transformation
    for (int i = 0, vidx = 0, jidx = 0; i<npts; i++) {
      float wpt = 1.0;

      // exclude mouth region and chin region
      if (!params.fitExpression) {
        if (i >= 48 || (i >= 4 && i <= 12))
          wpt = 0.9;
      }
      else if (i < 17) wpt = 1.0;

      // point p
      float px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      //cout << px << ", " << py << ", " << pz << endl;

      // point q
      const PhGUtils::Point2f& q = targets_2d[i].q;

      // R * p
      float Rpx = px, Rpy = py, Rpz = pz;
      PhGUtils::rotatePoint(Rpx, Rpy, Rpz, R);

      // P = R * p + t
      float Pkx = Rpx + tx, Pky = Rpy + ty, Pkz = Rpz + tz;

      // Jf
      float inv_z = 1.0 / Pkz;
      float inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Pkx * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Pky * inv_z2;

      /*
      Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
      */

      // project p to color image plane
      float pu, pv;
      PhGUtils::projectPoint(pu, pv, Pkx, Pky, Pkz, Mproj);

      // residue
      float rkx = pu - q.x, rky = pv - q.y;
      double rk = sqrt(rkx*rkx + rky*rky);
      double inv_rk = 1.0 / rk;
      // normalize it
      rkx *= inv_rk; rky *= inv_rk;

      // J_? * p_k
      float jpx, jpy, jpz;
      // J_f * J_? * p_k
      float jfjpx, jfjpy;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jx);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      /*
      cout << "jf\t";
      PhGUtils::printArray(Jf, 9);
      cout << "j\t" << jpx << ", " << jpy << ", "  << jpz << endl;
      cout << "jj\t" << jfjpx << ", " << jfjpy << ", "  << jfjpz << endl;
      */

      // \frac{\partial r_i}{\partial \theta_x}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jy);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      // \frac{\partial r_i}{\partial \theta_y}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jz);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      // \frac{\partial r_i}{\partial \theta_z}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt;

      // \frac{\partial r_i}{\partial \t_x}
      J[jidx++] = (Jf[0] * rkx) * wpt;

      // \frac{\partial r_i}{\partial \t_y}
      J[jidx++] = (Jf[4] * rky) * wpt;

      // \frac{\partial r_i}{\partial \t_z}
      J[jidx++] = (Jf[2] * rkx + Jf[5] * rky) * wpt;
    }

    /*
    ofstream fout("jacobian.txt");

    for(int i=0, jidx=0;i<npts;i++) {
    for(int j=0;j<7;j++) {
    fout << J[jidx++] << '\t';
    }
    fout << endl;
    }
    fout.close();


    ::system("pause");
    */
  }

  void cost_identity(float *p, float *hx, int m, int n, void* adata) {
    float s, rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto tm1cRT = model.tm1RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3f& T = params.Tvec;
    const PhGUtils::Matrix3x3f& R = params.Rmat;
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);

    //cout << "npts = " << npts << endl;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      float wpt = 1.0;

      float x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm1cRT(j, vidx) * p[j];
        y += tm1cRT(j, vidx + 1) * p[j];
        z += tm1cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      float u, v;
      PhGUtils::projectPoint(u, v, x + T.x, y + T.y, z + T.z, Mproj);

      float dx = q.x - u, dy = q.y - v;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx*dx + dy*dy);
      //cout << hx[i] << ", ";
    }

    // regularization term
    arma::fvec wmmu(m);
    for (int i = 0; i < m; ++i) {
      wmmu(i) = p[i] - params.mu_wid_orig(i);
    }
    hx[n - 1] = sqrt(arma::dot(wmmu, params.sigma_wid * wmmu)) * params.w_prior_id;
    //cout << hx[n] << endl;
  }

  void jacobian_identity(float *p, float *J, int m, int n, void* adata) {
    float rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto tm1cRT = model.tm1RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3f& T = params.Tvec;
    const PhGUtils::Matrix3x3f& R = params.Rmat;
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);

    //cout << "hx: ";
    float Jf[6] = { 0 };
    float f_x = params.camparams.fx, f_y = params.camparams.fy;

    int jidx = 0;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      float wpt = 1.0;

      // Pk
      float Px = 0, Py = 0, Pz = 0;
      for (int j = 0; j < m; j++) {
        Px += tm1cRT(j, vidx) * p[j];
        Py += tm1cRT(j, vidx + 1) * p[j];
        Pz += tm1cRT(j, vidx + 2) * p[j];
      }
      Px += T.x; Py += T.y; Pz += T.z;

      const PhGUtils::Point2f& q = targets_2d[i].q;
      float u, v;
      PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

      float rkx = u - q.x, rky = v - q.y;

      double rk = sqrt(rkx*rkx + rky*rky);
      double inv_rk = 1.0 / rk;
      rkx *= inv_rk; rky *= inv_rk;

      // Jf
      float inv_z = 1.0 / Pz;
      float inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

      for (int j = 0; j < m; ++j) {
        // R * Jm * e_j
        float jmex = tm1cRT(j, vidx), jmey = tm1cRT(j, vidx + 1), jmez = tm1cRT(j, vidx + 2);

        // Jf * R * Jm * e_j
        float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
        float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
        // data term and prior term
        J[jidx++] = (jfjmx * rkx + jfjmy * rky);
      }
    }

    // regularization term
    arma::fvec wmmu(m);
    for (int j = 0; j < m; ++j) {
      wmmu(j) = p[j] - params.mu_wid_orig(j);
    }
    arma::fvec Jprior = params.sigma_wid * wmmu;
    float Eprior = sqrt(arma::dot(wmmu, Jprior));
    float inv_Eprior = 1.0 / Eprior;

    for (int j = 0; j < m; ++j) {
      J[jidx++] = inv_Eprior * Jprior(j) * params.w_prior_id;
    }
  }

  void cost_expression(float *p, float *hx, int m, int n, void* adata) {
    float rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto tm0cRT = model.tm0RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3f& T = params.Tvec;
    const PhGUtils::Matrix3x3f& R = params.Rmat;
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);

    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      float wpt = 1.0;

      float x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm0cRT(j, vidx) * p[j];
        y += tm0cRT(j, vidx + 1) * p[j];
        z += tm0cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      float u, v;
      PhGUtils::projectPoint(u, v, x + T.x, y + T.y, z + T.z, Mproj);

      float dx = q.x - u, dy = q.y - v;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx*dx + dy*dy);
    }

    // regularization term
    int nparams = params.mu_wexp.n_elem;
    arma::fvec wmmu(nparams);
    for (int i = 0; i < nparams; ++i) {
      wmmu(i) = p[i] - params.mu_wexp_orig(i);
    }
    hx[n - 1] = sqrt(arma::dot(wmmu, params.sigma_wexp * wmmu)) * params.w_prior_exp;
  }

  void jacobian_expression(float *p, float *J, int m, int n, void* adata) {
    float rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto tm0cRT = model.tm0RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3f& T = params.Tvec;
    const PhGUtils::Matrix3x3f& R = params.Rmat;
    PhGUtils::Matrix3x3f Mproj(params.camparams.fx, 0, params.camparams.cx,
      0, params.camparams.fy, params.camparams.cy,
      0, 0, 1);

    float Jf[6] = { 0 };
    float f_x = params.camparams.fx, f_y = params.camparams.fy;

    // regularization term
    int nparams = params.mu_wexp_orig.n_elem;
    arma::fvec wmmu(nparams);
    for (int i = 0; i < nparams; ++i) {
      wmmu(i) = p[i] - params.mu_wexp_orig(i);
    }
    arma::fvec Jprior = 2.0 * params.sigma_wexp * wmmu;

    int jidx = 0;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      float wpt = 1.0;

      float x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm0cRT(j, vidx) * p[j];
        y += tm0cRT(j, vidx + 1) * p[j];
        z += tm0cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      float Px = x + T.x, Py = y + T.y, Pz = z + T.z;
      float u, v;
      PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

      float dx = u - q.x, dy = v - q.y;

      float rk = sqrt(dx*dx + dy*dy);
      float inv_rk = 1.0 / rk;

      // Jf
      float inv_z = 1.0 / Pz;
      float inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

      for (int j = 0; j < m; ++j) {
        // Jm * e_j
        float jmex = tm0cRT(j, vidx), jmey = tm0cRT(j, vidx + 1), jmez = tm0cRT(j, vidx + 2);
        // Jf * Jm * e_j
        float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
        float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
        // data term and prior term
        J[jidx++] = (jfjmx * dx + jfjmy * dy) * inv_rk;
      }
    }

    for (int j = 0; j < m; ++j) {
      J[jidx++] = Jprior(j) * params.w_prior_exp;
    }
  }

  fptr f_pose, jac_pose, f_id, jac_id, f_exp, jac_exp;

  MultilinearModel<float> &model;
  DefaultParameters &params;
  const vector<Constraint_2D> &targets_2d;
};
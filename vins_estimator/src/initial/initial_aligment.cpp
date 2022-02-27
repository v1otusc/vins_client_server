#include "initial_alignment.h"

/**
 * @brief 陀螺仪 Bias 矫正
 *        - 根据视觉 SFM 的结果来矫正陀螺仪 Bias -> vins 论文 V-B-1
 *          思路是将相邻图像帧之间 sfm 求解出来的旋转量与预积分旋转量对齐
 *        - 得到新的 bias 之后进行 repropagate
 * @param[in] all_image_frame所有图像帧构成的map
 * @param[out] Bgs矫正后的位姿
 * @return void
 */
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                        Vector3d *Bgs) {
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();

    // c0 指参考帧(l)
    // q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    // tmp_A = J_j_bw
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(
        O_R, O_BG);
    // tmp_B = 2 * (r^bk_bk+1)^-1 * q_ij
    tmp_b =
        2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    // tempA * delta_bg = tmp_b
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  // ldlt() Cholesky 分解的一种，满足 A 为正定矩阵
  // the Cholesky decomposition with full pivoting without square root of *this
  delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration "
                  << delta_bg.transpose());

  // 更新陀螺仪 bias
  for (int i = 0; i <= WINDOW_SIZE; i++) Bgs[i] += delta_bg;

  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    // Bias 更新，需要根据新的 Bias 重新计算预积分
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}

// 求重力切空间的两正交单位向量--3x1, 返回 3x2
MatrixXd TangentBasis(Vector3d &g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

/**
 * @brief 重力矢量细化
 *        在其切线空间上用两个参数重新参数化重力(两自由度) -> paper-V-B-3
 * @param[in] all_image_frame 所有时间戳-图像帧构成的 map
 * @param[out] g 重力加速度(在第0帧相机坐标系下)
 * @param[out] x 优化变量，all_image_frame 每帧速度v[0:n],二自由度g,尺度s
 * @return 是否细化成功
 */
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                   VectorXd &x) {
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  // 迭代求解四次
  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 *
                                Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() *
                                (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) =
          frame_j->second.pre_integration->delta_p +
          frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
          frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) =
          frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) =
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) =
          frame_j->second.pre_integration->delta_v -
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
  }
  g = g0;
}

/**
 * @brief 初始化速度，重力和尺度因子--所谓 linear 即求解线性最小二乘
 *        Paper -> V-B-2
 *        相邻图像帧之间的平移向量和速度与IMU预积分得到的对齐，求解线性最小二乘
 *        重力细化 -- Paper -> V-B-3
 * @param[in] all_image_frame 所有时间戳-图像帧构成的 map
 * @param[out] g 重力加速度
 * @param[out] x 优化变量，all_image_frame 每帧速度v[0:n],g,尺度s
 * @return  bool -- 是否线性初始化成功
 */
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                     VectorXd &x) {
  int all_frame_count = all_image_frame.size();
  // 待优化量维度
  int n_state = all_frame_count * 3 + 3 + 1;

  // Ax = b
  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    // @see 崔华坤 P19 推导
    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) =
        frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() *
                              (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) =
        frame_j->second.pre_integration->delta_p +
        frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];

    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) =
        frame_i->second.R.transpose() * dt * Matrix3d::Identity();

    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);

  // 从优化变量中取出尺度 s
  double s = x(n_state - 1) / 100.0;
  ROS_DEBUG("estimated scale: %f", s);
  // 取出对重力向量的计算值
  g = x.segment<3>(n_state - 4);
  ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if (fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
    return false;
  }

  // 重力细化
  RefineGravity(all_image_frame, g, x);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
  if (s < 0.0)
    return false;
  else
    return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs,
                        Vector3d &g, VectorXd &x) {
  // 计算陀螺仪偏置--利用旋转约束
  solveGyroscopeBias(all_image_frame, Bgs);

  // 计算尺度，refined-重力加速度和速度--利用平移约束
  if (LinearAlignment(all_image_frame, g, x))
    return true;
  else
    return false;
}

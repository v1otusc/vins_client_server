#include "initial_sfm.h"

GlobalSFM::GlobalSFM() {}

/**
 * @brief 三角化两帧之间某个对应点深度
 *        线性三角化法DLT
 * @param[in] 位姿和归一化坐标
 *            poit0 - Pose0 位姿(相对于 l 帧)对应的特征点
 * @param[out] 3d landmark 点坐标(point_3d)
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                 Eigen::Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1,
                                 Vector3d &point_3d) {
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * @brief PnP 方法得到l帧到i帧的位姿
 * @param[in] sfm_f
 * @param[out] R_initial
 * @param[out] P_initial
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_f) {
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;
  for (int j = 0; j < feature_num; j++) {
    // 没有三角化, continue
    if (sfm_f[j].state != true) continue;
    Vector2d point2d;
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      if (sfm_f[j].observation[k].first == i) {
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                          sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }
  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10) return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}

/**
 * @brief 三角化 frame0 和 frame1 之间所有对应点
 * @param[in] frame, Pose, sfm_f(用到帧索引和特征点归一化坐标)
 * @param[out] sfm_f 的 state 和三角化出的 3d 坐标
 */
void GlobalSFM::triangulateTwoFrames(int frame0,
                                     Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1,
                                     Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f) {
  assert(frame0 != frame1);
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true) continue;
    bool has_0 = false, has_1 = false;
    Vector2d point0;
    Vector2d point1;
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      if (sfm_f[j].observation[k].first == frame0) {
        point0 = sfm_f[j].observation[k].second;
        has_0 = true;
      }
      if (sfm_f[j].observation[k].first == frame1) {
        point1 = sfm_f[j].observation[k].second;
        has_1 = true;
      }
    }
    if (has_0 && has_1) {
      Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
    }
  }
}

/**
 * @brief 纯视觉 sfm, 求解窗口中所有图像帧相对于l帧的位姿和三角化的特征点坐标
 * @param[in] frame_num=frame_count+1, 索引方便
 * @param[out] q, 窗口内图像帧相对于l帧的旋转四元数q[frame_count+1]
 * @param[out] T, 窗口内图像帧相对于l帧的平移向量T[frame_count+1]
 * @param[in] l, 第l帧
 * @param[in] relative_R, 最新帧到第l帧的旋转矩阵
 * @param[in] relative_T, 最新帧到第l帧的平移向量
 * @param[in] sfm_f 所有特征点
 * @param[out] sfm_tracked_points 所有在sfm中三角化的特征点id和3d坐标
 * @return
 */
bool GlobalSFM::construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                          const Matrix3d relative_R, const Vector3d relative_T,
                          vector<SFMFeature> &sfm_f,
                          map<int, Vector3d> &sfm_tracked_points) {
  feature_num = sfm_f.size();
  // cout << "set 0 and " << l << " as known " << endl;
  // have relative_r relative_t
  // intial two view
  // 首先是l帧和最新帧--放入q,T对应位置
  q[l].w() = 1;
  q[l].x() = 0;
  q[l].y() = 0;
  q[l].z() = 0;
  T[l].setZero();
  q[frame_num - 1] = q[l] * Quaterniond(relative_R);
  T[frame_num - 1] = relative_T;
  // cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
  // cout << "init t_l " << T[l].transpose() << endl;

  // l frame rotate to other cam frame
  Matrix3d c_Rotation[frame_num];
  Vector3d c_Translation[frame_num];
  Quaterniond c_Quat[frame_num];
  double c_rotation[frame_num][4];
  double c_translation[frame_num][3];
  // 位姿3x4,l帧到每一帧的变换矩阵
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  c_Quat[l] = q[l].inverse();
  c_Rotation[l] = c_Quat[l].toRotationMatrix();
  c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
  Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
  Pose[l].block<3, 1>(0, 3) = c_Translation[l];

  c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
  c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
  c_Translation[frame_num - 1] =
      -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
  Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
  Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

  // 1: trangulate between l ----- frame_num - 1
  // 2: solve pnp l + 1, trangulate l + 1 ------- frame_num - 1;
  for (int i = l; i < frame_num - 1; i++) {
    // solve pnp
    // 不是参考帧,先用pnp求解位姿
    if (i > l) {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }
  // 3: triangulate l-----l+1 l+2 ... frame_num -2
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
  // 4: solve pnp l-1; triangulate l-1 ----- l
  //              l-2              l-2 ----- l
  //              ...              ...
  for (int i = l - 1; i >= 0; i--) {
    // solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    // triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }
  // 5: triangulate all other points
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true) continue;
    if ((int)sfm_f[j].observation.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
    }
  }

  // full BA--3d landmark 和特征点归一化坐标
  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new ceres::QuaternionParameterization();

  for (int i = 0; i < frame_num; i++) {
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);
    if (i == l) {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1) {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  for (int i = 0; i < feature_num; i++) {
    if (sfm_f[i].state != true) continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
      int l = sfm_f[i].observation[j].first;
      ceres::CostFunction *cost_function =
          ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                      sfm_f[i].observation[j].second.y());

      problem.AddResidualBlock(cost_function, NULL, c_rotation[l],
                               c_translation[l], sfm_f[i].position);
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (summary.termination_type == ceres::CONVERGENCE ||
      summary.final_cost < 5e-03) {
    // cout << "vision only BA converge" << endl;
  } else {
    // cout << "vision only BA not converge " << endl;
    return false;
  }
  for (int i = 0; i < frame_num; i++) {
    q[i].w() = c_rotation[i][0];
    q[i].x() = c_rotation[i][1];
    q[i].y() = c_rotation[i][2];
    q[i].z() = c_rotation[i][3];
    q[i] = q[i].inverse();
  }
  for (int i = 0; i < frame_num; i++) {
    T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1],
                                 c_translation[i][2]));
  }
  for (int i = 0; i < (int)sfm_f.size(); i++) {
    if (sfm_f[i].state)
      sfm_tracked_points[sfm_f[i].id] = Vector3d(
          sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
  }
  return true;
}

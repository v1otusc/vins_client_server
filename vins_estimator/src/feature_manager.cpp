#include "feature_manager.h"

int FeaturePerId::endFrame() {
  return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[]) : Rs(_Rs) {
  for (int i = 0; i < NUM_OF_CAM; i++) ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState() { feature.clear(); }

int FeatureManager::getFeatureCount() {
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
      cnt++;
    }
  }
  return cnt;
}

/**
 * @brief 通过检测两帧之间的视差以及特征点数量决定次新帧是否是关键帧
 * @param frame_count 滑窗中帧的数量
 * @param image 输入图像中的特征点信息
 * @param td IMU 和相机同步时间差
 * @return true 则删除最旧的帧 false 则删除次新帧
 */
bool FeatureManager::addFeatureCheckParallax(
    int frame_count,
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    double td) {
  // 输入的特征点数量
  ROS_DEBUG("input feature: %d", (int)image.size());
  // 能够作为特征点的数量
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;
  int parallax_num = 0;
  // 此图像帧中被跟踪的特征点数量
  last_track_num = 0;
  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

    int feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return it.feature_id == feature_id;
                      });

    if (it == feature.end()) {
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(f_per_fra);
    } else if (it->feature_id == feature_id) {
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++;
    }
  }

  // 追踪次数小于 20 或窗口内帧的数目小于 2，是关键帧
  if (frame_count < 2 || last_track_num < 20) return true;

  // 计算每个特征在次新帧(倒数第二帧)和次次新帧(倒数第三帧)中的视差
  for (auto &it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >=
            frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) {
    return true;
  } else {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum,
              parallax_num);
    ROS_DEBUG("current parallax: %lf",
              parallax_sum / parallax_num * FOCAL_LENGTH);
    // 平均视差大于阈值的关键帧被保留
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

void FeatureManager::debugShow() {
  ROS_DEBUG("debug show");
  for (auto &it : feature) {
    ROS_ASSERT(it.feature_per_frame.size() != 0);
    ROS_ASSERT(it.start_frame >= 0);
    ROS_ASSERT(it.used_num >= 0);

    ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
    int sum = 0;
    for (auto &j : it.feature_per_frame) {
      ROS_DEBUG("%d,", int(j.is_used));
      sum += j.is_used;
      printf("(%lf,%lf) ", j.point(0), j.point(1));
    }
    ROS_ASSERT(it.used_num == sum);
  }
}

// 得到 frame_count_l 与 frame_count_r 两帧之间的对应特征点归一化坐标
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(
    int frame_count_l, int frame_count_r) {
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature) {
    // 要找特征点的两帧在 it.start_frame~it.endFrame() 窗口范围内
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;
      a = it.feature_per_frame[idx_l].point;
      b = it.feature_per_frame[idx_r].point;
      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

// 设置特征点的逆深度估计值
void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    } else
      it_per_id.solve_flag = 1;
  }
}

void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2) feature.erase(it);
  }
}

void FeatureManager::clearDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth = 1.0 / x(++feature_index);
  }
}

/**
 * @brief 窗口中所有被跟踪的特征点的逆深度组成的 vector
 * @return VectorXd
 */
VectorXd FeatureManager::getDepthVector() {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
  }
  return dep_vec;
}

// 对特征点进行三角化求深度（SVD分解）
// (而不是 initial_sfm.cpp 的直接法)
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[],
                                 Matrix3d ric[]) {
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    if (it_per_id.estimated_depth > 0) continue;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    ROS_ASSERT(NUM_OF_CAM == 1);
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j) continue;
    }
    ROS_ASSERT(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V =
        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
            .matrixV()
            .rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    it_per_id.estimated_depth = svd_method;
    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = INIT_DEPTH;
    }
  }
}

void FeatureManager::removeOutlier() {
  ROS_BREAK();
  int i = -1;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    i += it->used_num != 0;
    if (it->used_num != 0 && it->is_outlier == true) {
      feature.erase(it);
    }
  }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                          Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      } else {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = INIT_DEPTH;
      }
    }
  }
}

void FeatureManager::removeBack() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1) continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

// 计算相邻两个关键帧的视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
  // check the second last frame is keyframe or not
  // parallax betwwen seconde last frame and third last frame
  const FeaturePerFrame &frame_i =
      it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j =
      it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans = 0;
  Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  Vector3d p_i_comp;
  p_i_comp = p_i;
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  ans = max(
      ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}
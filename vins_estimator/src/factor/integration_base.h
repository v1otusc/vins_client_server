#pragma once

#include "../parameters.h"
#include "../utility/utility.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationBase {
 public:
  IntegrationBase() = delete;
  IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                  const Eigen::Vector3d &_linearized_ba,
                  const Eigen::Vector3d &_linearized_bg)
      : acc_0{_acc_0},
        gyr_0{_gyr_0},
        linearized_acc{_acc_0},
        linearized_gyr{_gyr_0},
        linearized_ba{_linearized_ba},
        linearized_bg{_linearized_bg},
        jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
        covariance{Eigen::Matrix<double, 15, 15>::Zero()},
        sum_dt{0.0},
        delta_p{Eigen::Vector3d::Zero()},
        delta_q{Eigen::Quaterniond::Identity()},
        delta_v{Eigen::Vector3d::Zero()}

  {
    noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
  }

  void push_back(double dt, const Eigen::Vector3d &acc,
                 const Eigen::Vector3d &gyr) {
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);
    // 进入预积分--增量PVQ传播公式
    propagate(dt, acc, gyr);
  }

  // 优化过程中Bias会更新，需要根据新的bias重新计算预积分
  void repropagate(const Eigen::Vector3d &_linearized_ba,
                   const Eigen::Vector3d &_linearized_bg) {
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
      propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
  }

  /**
   * @brief 离散形式的中值积分
   *        - 参数中 _0 代表上次测量值 _1 代表当前测量值
   *        - delta_p, delta_q, delta_v 代表相对预积分初始参考帧的PVQ
   *        - 从第 k 帧预积分到 k + 1 帧，则参考系是 k 帧的 imu 坐标系
   */
  void midPointIntegration(
      double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
      const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
      const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
      const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
      const Eigen::Vector3d &linearized_bg, Eigen::Vector3d &result_delta_p,
      Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
      Eigen::Vector3d &result_linearized_ba,
      Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
    // ROS_INFO("midpoint integration");
    // delta_q 为相对预积分参考系的旋转四元数 -- q_i
    // --其结果是线加速度从世界坐标系转化到了 b_k
    Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
    // 计算平均角速度，不必左乘 delta_q
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    // 平均角速度和时间的乘机构成的旋转值
    // w = 1/2 * (w_i + w_{i+1}) - bg
    // 左乘旋转四元数，获得当前时刻到参考帧k body 坐标系的旋转向量 q_{i+1}
    result_delta_q =
        delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2,
                              un_gyr(2) * _dt / 2);
    // 用计算出来的旋转向量左乘，将线加速度从世界坐标系下转到了 body 坐标系下
    Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
    // 平均线加速度
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 当前(i+1) delta_p -- 以 b_k 为坐标系的增量信息
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    // 当前(i+1) delta_v -- 以 b_k 为坐标系的增量信息
    result_delta_v = delta_v + un_acc * _dt;

    // 预积分过程中 Bias 并未发生改变，所以还保存在 result 当中
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    // 更新协方差矩阵以及 jacobian 矩阵
    if (update_jacobian) {
      // 计算平均角速度
      Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
      // 减去 bias 的加速度
      Vector3d a_0_x = _acc_0 - linearized_ba;
      Vector3d a_1_x = _acc_1 - linearized_ba;
      Matrix3d R_w_x, R_a_0_x, R_a_1_x;

      R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
      R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
          a_0_x(0), 0;
      R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
          a_1_x(0), 0;

      MatrixXd F = MatrixXd::Zero(15, 15);
      F.block<3, 3>(0, 0) = Matrix3d::Identity();
      F.block<3, 3>(0, 3) =
          -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
          -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
              (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
      F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(0, 9) =
          -0.25 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          _dt * _dt;
      F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() *
                             R_a_1_x * _dt * _dt * -_dt;
      F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
      F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                                (Matrix3d::Identity() - R_w_x * _dt) * _dt;
      F.block<3, 3>(6, 6) = Matrix3d::Identity();
      F.block<3, 3>(6, 9) =
          -0.5 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          _dt;
      F.block<3, 3>(6, 12) =
          -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
      F.block<3, 3>(9, 9) = Matrix3d::Identity();
      F.block<3, 3>(12, 12) = Matrix3d::Identity();
      // cout<<"A"<<endl<<A<<endl;

      MatrixXd V = MatrixXd::Zero(15, 18);
      V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() *
                            R_a_1_x * _dt * _dt * 0.5 * _dt;
      V.block<3, 3>(0, 6) =
          0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
      V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 3) =
          0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
      V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
      V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

      // step_jacobian = F;
      // step_V = V;
      // jacobian 矩阵的迭代公式: J_(k+1) = F*J_K, J_0 = I
      jacobian = F * jacobian;
      // covariance 的迭代公式为：P_(k+1) = F*P_K*F^T + V*Q*V^T, P_0 = 0
      // P_k 为协方差，Q 为 noise，初值为 18*18 的单位矩阵
      covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
  }

  /**
   * @brief 根据两imu帧之间PVQ相对于参考帧增量的关系式，进行增量PVQ的传播
   *        -- 所谓的传播公式
   *        -- 各种论文推导中表达的是：两帧之间PVQ增量的中值法离散形式
   */
  void propagate(double _dt, const Eigen::Vector3d &_acc_1,
                 const Eigen::Vector3d &_gyr_1) {
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    // 中值积分
    midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q,
                        delta_v, linearized_ba, linearized_bg, result_delta_p,
                        result_delta_q, result_delta_v, result_linearized_ba,
                        result_linearized_bg, 1);

    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
  }

  /**
   * @brief 计算 IMU 的残差
   *        两图像帧 bk - bk+1 之间 PVQ 和 bias 的变化量的差
   * @param[in] Pi第i帧位置
   * @param[in] Qi第i帧旋转
   * @param[in] Vi第i帧速度
   * @param[in] Bai第i帧加速度Bias
   * @param[in] Bgi第i帧角速度Bias
   * @param[in] pj第j帧位置
   * @param[in] Qj第j帧旋转
   * @param[in] Vj第j帧速度
   * @param[in] Baj第j帧加速度Bias
   * @param[in] Bgj第j帧角速度Bias
   * @return
   */
  Eigen::Matrix<double, 15, 1> evaluate(
      const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
      const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
      const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
      const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
      const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
    Eigen::Matrix<double, 15, 1> residuals;

    Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbg = Bgi - linearized_bg;

    Eigen::Quaterniond corrected_delta_q =
        delta_q * Utility::deltaQ(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(O_P, 0) =
        Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) -
        corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) =
        2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) =
        Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
  }

  double dt;
  Eigen::Vector3d acc_0, gyr_0;
  Eigen::Vector3d acc_1, gyr_1;

  const Eigen::Vector3d linearized_acc, linearized_gyr;
  Eigen::Vector3d linearized_ba, linearized_bg;

  Eigen::Matrix<double, 15, 15> jacobian, covariance;
  Eigen::Matrix<double, 15, 15> step_jacobian;
  Eigen::Matrix<double, 15, 18> step_V;
  Eigen::Matrix<double, 18, 18> noise;

  double sum_dt;
  Eigen::Vector3d delta_p;
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_v;

  std::vector<double> dt_buf;
  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;
};
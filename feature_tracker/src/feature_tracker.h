#pragma once

#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker {
 public:
  FeatureTracker();

  void readImage(const cv::Mat &_img, double _cur_time);

  void setMask();

  void addPoints();

  bool updateID(unsigned int i);

  void readIntrinsicParameter(const string &calib_file);

  void showUndistortion(const string &name);

  void rejectWithF();

  void undistortedPoints();

  // 图像掩码
  cv::Mat mask;
  // 鱼眼掩码，用来去除边缘噪点
  cv::Mat fisheye_mask;
  // 上一次发布的帧的图像数据
  cv::Mat prev_img;
  // 光流跟踪的前一帧的图像数据
  cv::Mat cur_img;
  // 光流跟踪的后一帧的图像数据
  cv::Mat forw_img;
  // 每一帧中新提取的特征点
  vector<cv::Point2f> n_pts;
  // 对应帧的图像中的特征点的像素坐标
  vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  // 去畸变的归一化坐标系的点，z==1 只是记录二维
  vector<cv::Point2f> prev_un_pts, cur_un_pts;
  // 当前帧相对前一帧特征点沿 x, y 方向的相机移动速度
  vector<cv::Point2f> pts_velocity;
  // 能够被追踪到的特征点的 id
  vector<int> ids;
  // 当前帧 forw_img 中每个特征点被追踪的时间次数
  vector<int> track_cnt;
  // id-特征点 map
  map<int, cv::Point2f> cur_un_pts_map;
  map<int, cv::Point2f> prev_un_pts_map;
  // 相机模型
  camodocal::CameraPtr m_camera;
  double cur_time;
  double prev_time;
  // 用来作为特征点 id，每检测到一个新的特征点，就将 n_id 作为该特征点的 id,
  // 并且 n_id++
  static int n_id;
};

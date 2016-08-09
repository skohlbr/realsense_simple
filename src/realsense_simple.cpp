#include <librealsense/rs.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <sstream>

#include <realsense_simple/realsense_simple.h>

void Realsense::streamToImage(cv::Mat* image, rs::stream stream) {
  rs::format format = dev_->get_stream_format(stream);

  int imgTypeInt;

  switch (format) {
    case rs::format::z16:
      readRawData<uint16_t>(image, stream, CV_16UC1);
      break;
    case rs::format::rgb8:
      readRawData<color>(image, stream, CV_8UC3);
      break;
    case rs::format::y8:
      readRawData<uint8_t>(image, stream, CV_8UC1);
      break;
    default:
      ROS_FATAL("Unsupported image format");
      exit(EXIT_FAILURE);
  }
}

void Realsense::filterDepth() {
  if (enable_median_filter_) {
    cv::medianBlur(depth_image_, depth_image_, 5);
  }

  cv::Mat max_image(depth_image_.rows, depth_image_.cols, CV_16UC1);
  cv::Mat min_image(depth_image_.rows, depth_image_.cols, CV_16UC1);

  if (enable_min_max_filter_) {
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(min_max_filter_size_, min_max_filter_size_));
    cv::erode(depth_image_, min_image, kernel);
    cv::dilate(depth_image_, max_image, kernel);

    for (size_t i = 0; i < depth_image_.rows * depth_image_.cols; ++i) {
      float diff = (reinterpret_cast<uint16_t*>(max_image.data)[i] -
                    reinterpret_cast<uint16_t*>(min_image.data)[i]) *
                   dev_->get_depth_scale();

      if (diff > min_max_filter_threshold_) {
        reinterpret_cast<uint16_t*>(depth_image_.data)[i] = 0;
      }
    }
  }
}

void Realsense::downSampleDepth() {
  // count valid samples
  int num_valid_samples = 0;
  for (size_t i = 0; i < depth_image_.rows * depth_image_.cols; ++i) {
    if (reinterpret_cast<uint16_t*>(depth_image_.data)[i] != 0) {
      ++num_valid_samples;
    }
  }

  // return if there is no need for further down sampling
  if (num_valid_samples <= target_number_of_samples_) {return;}

  double skip = static_cast<double>(num_valid_samples) /
                (num_valid_samples - target_number_of_samples_);

  double f_idx = 0.0;
  int idx = 0;

  while (idx < depth_image_.rows * depth_image_.cols) {
    reinterpret_cast<uint16_t*>(depth_image_.data)[idx] = 0;
    f_idx += skip;
    idx = static_cast<int>(f_idx);
  }
}

void Realsense::buildNewPointcloud() {
  if (enable_pointcloud_color_) {
    pointcloud_color_.clear();
  }
  if (enable_pointcloud_no_intensity_) {
    pointcloud_no_intensity_.clear();
  }

  for (size_t y_pixels = 0; y_pixels < depth_image_.rows; ++y_pixels) {
    for (size_t x_pixels = 0; x_pixels < depth_image_.cols; ++x_pixels) {
      rs::float2 depth_coord;
      depth_coord.x = x_pixels;
      depth_coord.y = y_pixels;

      size_t depth_idx = x_pixels + depth_image_.cols * y_pixels;

      float z = reinterpret_cast<uint16_t*>(depth_image_.data)[depth_idx] *
                dev_->get_depth_scale();

      if (z > 0.0) {
        rs::float3 point = depth_intrin_.deproject(depth_coord, z);

        if (enable_pointcloud_color_) {
          color image_element;
          pcl::PointXYZRGB point_out;
          if (getImageElementRef<color>(&image_element, color_camera_extrin_,
                                        color_camera_intrin_, point,
                                        color_image_)) {
            point_out.x = point.x;
            point_out.y = point.y;
            point_out.z = point.z;
            point_out.r = image_element.r;
            point_out.g = image_element.g;
            point_out.b = image_element.b;

            pointcloud_color_.push_back(point_out);
          }
        }
        if (enable_pointcloud_no_intensity_) {
          pointcloud_no_intensity_.push_back(
              pcl::PointXYZ(point.x, point.y, point.z));
        }
      }
    }
  }
}

void Realsense::getCameraInfo(sensor_msgs::CameraInfo* camera_info,
                              const std::string& frame_id,
                              const rs::intrinsics& intrin,
                              const rs::extrinsics& extrin) {
  // create color camera calibration properties message
  camera_info->header.stamp = ros::Time::now();
  camera_info->header.frame_id = frame_id;
  camera_info->height = intrin.height;
  camera_info->width = intrin.width;
  camera_info->distortion_model = "plumb_bob";

  camera_info->D.resize(5);
  camera_info->D[0] = intrin.coeffs[0];
  camera_info->D[1] = intrin.coeffs[1];
  camera_info->D[2] = intrin.coeffs[2];
  camera_info->D[3] = intrin.coeffs[3];
  camera_info->D[4] = intrin.coeffs[4];

  camera_info->K[0] = intrin.fx;
  camera_info->K[1] = 0.0;
  camera_info->K[2] = intrin.ppx;
  camera_info->K[3] = 0.0;
  camera_info->K[4] = intrin.fy;
  camera_info->K[5] = intrin.ppy;
  camera_info->K[6] = 0.0;
  camera_info->K[7] = 0.0;
  camera_info->K[8] = 1.0;

  camera_info->R[0] = 1.0;
  camera_info->R[1] = 0.0;
  camera_info->R[2] = 0.0;
  camera_info->R[3] = 0.0;
  camera_info->R[4] = 1.0;
  camera_info->R[5] = 0.0;
  camera_info->R[6] = 0.0;
  camera_info->R[7] = 0.0;
  camera_info->R[8] = 1.0;

  /*camera_info->P[0] =
      intrin.fx * extrin.rotation[0] + intrin.ppx * extrin.rotation[2];
  camera_info->P[1] =
      intrin.fx * extrin.rotation[3] + intrin.ppx * extrin.rotation[5];
  camera_info->P[2] =
      intrin.fx * extrin.rotation[6] + intrin.ppx * extrin.rotation[8];
  camera_info->P[3] =
      intrin.fx * extrin.translation[0] + intrin.ppx * extrin.translation[2];
  camera_info->P[4] =
      intrin.fy * extrin.rotation[1] + intrin.ppy * extrin.rotation[2];
  camera_info->P[5] =
      intrin.fy * extrin.rotation[4] + intrin.ppy * extrin.rotation[5];
  camera_info->P[6] =
      intrin.fy * extrin.rotation[7] + intrin.ppy * extrin.rotation[8];
  camera_info->P[7] =
      intrin.fy * extrin.translation[1] + intrin.ppy * extrin.translation[2];
  camera_info->P[8] = extrin.rotation[2];
  camera_info->P[9] = extrin.rotation[5];
  camera_info->P[10] = extrin.rotation[8];
  camera_info->P[11] = extrin.translation[2];*/

  camera_info->P[0] = intrin.fx;
  camera_info->P[1] = 0;
  camera_info->P[2] = intrin.ppx;
  camera_info->P[3] = 0;
  camera_info->P[4] = 0;
  camera_info->P[5] = intrin.fy;
  camera_info->P[6] = intrin.ppy;
  camera_info->P[7] = 0;
  camera_info->P[8] = 0;
  camera_info->P[9] = 0;
  camera_info->P[10] = 1;
  camera_info->P[11] = 0;
}

void Realsense::setupCalibration() {
  if (enable_color_camera_info_ || enable_pointcloud_color_) {
    color_camera_extrin_ =
        dev_->get_extrinsics(rs::stream::depth, rs::stream::color);
    color_camera_intrin_ = dev_->get_stream_intrinsics(rs::stream::color);
  }

  if (enable_depth_image_ || enable_pointcloud_color_ ||
      enable_pointcloud_no_intensity_) {
    depth_intrin_ = dev_->get_stream_intrinsics(rs::stream::depth);
  }
}

void Realsense::startDataStreams() {
  if (enable_color_image_ || enable_color_camera_info_ ||
      enable_pointcloud_color_) {
    dev_->enable_stream(rs::stream::color, rs::preset::best_quality);
  }
  if (enable_depth_image_ || enable_pointcloud_color_ ||
      enable_pointcloud_no_intensity_) {
    dev_->enable_stream(rs::stream::depth, rs::preset::best_quality);
  }
  dev_->start();
}

void Realsense::sendRosMessages() {
  if (enable_pointcloud_color_) {
    sensor_msgs::PointCloud2 points_msg;
    pcl::toROSMsg(pointcloud_color_, points_msg);
    points_msg.header.frame_id = "realsense/depth";
    points_msg.header.stamp = ros::Time::now();
    pointcloud_color_pub_.publish(points_msg);
  }
  if (enable_pointcloud_no_intensity_) {
    sensor_msgs::PointCloud2 points_msg;
    pcl::toROSMsg(pointcloud_no_intensity_, points_msg);
    points_msg.header.frame_id = "realsense/depth";
    points_msg.header.stamp = ros::Time::now();
    pointcloud_no_intensity_pub_.publish(points_msg);
  }

  if (enable_color_image_) {
    sensor_msgs::ImagePtr color_image_msg =
        cv_bridge::CvImage(std_msgs::Header(), "rgb8", color_image_)
            .toImageMsg();
    color_image_msg->header.frame_id = "realsense/color_camera";
    color_image_msg->header.stamp = ros::Time::now();
    color_image_pub_.publish(color_image_msg);
  }
  if (enable_depth_image_) {
    sensor_msgs::ImagePtr depth_image_msg =
        cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_image_)
            .toImageMsg();
    depth_image_msg->header.frame_id = "realsense/depth";
    depth_image_msg->header.stamp = ros::Time::now();
    depth_image_pub_.publish(depth_image_msg);
  }

  if (enable_color_camera_info_) {
    color_camera_info_pub_.publish(color_camera_info_);
  }
}

void Realsense::setupRos() {
  // get ros parameters
  private_nh_.param("enable_color_image", enable_color_image_,
                    kDefaultEnableColorImage);
  private_nh_.param("enable_color_camera_info", enable_color_camera_info_,
                    kDefaultEnableColorCameraInfo);
  private_nh_.param("enable_depth_image", enable_depth_image_,
                    kDefaultEnableDepthImage);
  private_nh_.param("enable_pointcloud_color", enable_pointcloud_color_,
                    kDefaultEnablePointcloudColor);
  private_nh_.param("enable_pointcloud_no_intensity_",
                    enable_pointcloud_no_intensity_,
                    kDefaultEnablePointcloudNoIntensity);

  private_nh_.param("downsample_rate_factor", downsample_rate_factor_,
                    kDefaultDownsampleRateFactor);
  private_nh_.param("target_number_of_samples", target_number_of_samples_,
                    kDefaultTargetNumberOfSamples);
  private_nh_.param("enable_median_filter", enable_median_filter_,
                    kDefaultEnableMedianFilter);
  private_nh_.param("enable_min_max_filter", enable_min_max_filter_,
                    kDefaultEnableMinMaxFilter);
  private_nh_.param("min_max_filter_size", min_max_filter_size_,
                    kDefaultMinMaxFilterSize);
  private_nh_.param("min_max_filter_threshold", min_max_filter_threshold_,
                    kDefaultMinMaxFilterThreshold);

  // setup publishers
  if (enable_color_image_) {
    color_image_pub_ = it_.advertise("color_image", 1);
  }
  if (enable_color_camera_info_) {
    color_camera_info_pub_ =
        private_nh_.advertise<sensor_msgs::CameraInfo>("color_camera_info", 1);
  }
  if (enable_depth_image_) {
    depth_image_pub_ = it_.advertise("depth_image", 1);
  }
  if (enable_pointcloud_color_) {
    pointcloud_color_pub_ =
        private_nh_.advertise<sensor_msgs::PointCloud2>("pointcloud_color", 1);
  }
  if (enable_pointcloud_no_intensity_) {
    pointcloud_no_intensity_pub_ =
        private_nh_.advertise<sensor_msgs::PointCloud2>(
            "pointcloud_no_intensity", 1);
  }
}

void Realsense::blockTillNextFrame() {
  size_t skip_frame = 0;

  while (skip_frame < downsample_rate_factor_) {
    if (dev_->is_streaming()) {
      dev_->wait_for_frames();
    }

    ++skip_frame;
  }
}

rs::device* Realsense::setupDevice(int argc, char* argv[]) {
  ros::init(argc, argv, "realsense_simple");
  rs::log_to_console(rs::log_severity::warn);

  if (ctx_.get_device_count() == 0) {
    ROS_FATAL("No device detected. Is it plugged in?");
    exit(EXIT_FAILURE);
  }

  return ctx_.get_device(0);
}

Realsense::Realsense(int argc, char* argv[])
    : dev_(setupDevice(argc, argv)), private_nh_("~"), it_(private_nh_) {
  setupRos();
  startDataStreams();
  setupCalibration();
}

void Realsense::processData() {
  if (enable_color_image_ || enable_pointcloud_color_) {
    streamToImage(&color_image_, rs::stream::color);
  }
  if (enable_depth_image_ || enable_pointcloud_color_ ||
      enable_pointcloud_no_intensity_) {
    streamToImage(&depth_image_, rs::stream::depth);
    filterDepth();
    if (target_number_of_samples_ > 0) {downSampleDepth();}
  }

  if (enable_pointcloud_color_ || enable_pointcloud_no_intensity_) {
    buildNewPointcloud();
  }

  if (enable_color_camera_info_) {
    getCameraInfo(&color_camera_info_, "realsense/color_camera",
                  color_camera_intrin_, color_camera_extrin_);
  }

  sendRosMessages();

  ros::spinOnce();
}

bool Realsense::ok(){
  return !ros::isShuttingDown();
}

int main(int argc, char* argv[]) {
  try {
    Realsense realsense(argc, argv);

    // start grabing and processing realsense data
    while (realsense.ok()) {
      realsense.blockTillNextFrame();
      realsense.processData();
    }

  } catch (const rs::error& e) {
    ROS_FATAL_STREAM("RealSense error calling " << e.get_failed_function()
                                                << "(" << e.get_failed_args()
                                                << "):\n    " << e.what());
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM(e.what());
  }
}

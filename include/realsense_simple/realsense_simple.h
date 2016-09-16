class Realsense {
 public:
  Realsense(int argc, char* argv[]);

  void blockTillNextFrame();

  void processData();

  bool ok();

  void shutdown();

 private:
  struct rgb_color {
    uint8_t r, g, b;
  };
  struct hsv_color {
    uint8_t h, s, v;
  };

  template <typename DataType>
  void readRawData(cv::Mat* image, const rs::stream stream, int cvImgType) {
    const DataType* data =
        reinterpret_cast<const DataType*>(dev_->get_frame_data(stream));
    *image = cv::Mat(dev_->get_stream_height(stream),
                     dev_->get_stream_width(stream), cvImgType);
    for (size_t i = 0; i < image->cols * image->rows; ++i) {
      reinterpret_cast<DataType*>(image->data)[i] = data[i];
    }
  }

  template <typename DataType>
  bool getImageElementRef(DataType* element, const rs::extrinsics& extrin,
                          const rs::intrinsics& intrin, const rs::float3& point,
                          const cv::Mat& image) {
    rs::float2 image_coord = intrin.project(extrin.transform(point));

    if ((image_coord.x < image.cols) && (image_coord.x >= 0) &&
        (image_coord.y < image.rows) && (image_coord.y >= 0)) {
      size_t image_idx =
          std::round(image_coord.x) + image.cols * std::round(image_coord.y);

      *element = reinterpret_cast<DataType*>(image.data)[image_idx];
      return true;
    } else {
      return false;
    }
  }

  void streamToImage(cv::Mat* image, rs::stream stream);

  void filterDepth();

  // set some of the depth values to 0 (invalid measurement) to approximately
  // match the desired number of samples
  void downSampleDepth();

  void buildNewPointcloud();

  void getCameraInfo(sensor_msgs::CameraInfo* camera_info,
                     const std::string& frame_id, const rs::intrinsics& intrin,
                     const rs::extrinsics& extrin);

  void setupCalibration();

  void startDataStreams();

  void sendRosMessages();

  void setupRos();

  rs::device* setupDevice(int argc, char* argv[]);

  // realsense device
  rs::context ctx_;
  rs::device* dev_;

  // ros handles
  ros::NodeHandle private_nh_;
  image_transport::ImageTransport it_;

  // ros publishers
  image_transport::Publisher color_image_pub_;
  ros::Publisher color_camera_info_pub_;
  image_transport::Publisher depth_image_pub_;
  ros::Publisher pointcloud_color_pub_;
  ros::Publisher pointcloud_no_intensity_pub_;

  // enable outputs
  bool enable_color_image_;
  bool enable_color_camera_info_;
  bool enable_depth_image_;
  bool enable_pointcloud_color_;
  bool enable_pointcloud_no_intensity_;

  // output rate
  int downsample_rate_factor_;
  // desired number of points for spatially down sampled point cloud
  int target_number_of_samples_;

  // depth filtering options
  bool enable_median_filter_;
  bool enable_min_max_filter_;
  int min_max_filter_size_;
  float min_max_filter_threshold_;

  // color filtering options
  bool enable_color_filter_;
  hsv_color hsv_min_;
  hsv_color hsv_max_;

  // camera calibrations
  rs::extrinsics color_camera_extrin_;

  rs::intrinsics depth_intrin_;
  rs::intrinsics color_camera_intrin_;

  // realsense data;
  cv::Mat rgb_color_image_;
  cv::Mat hsv_color_image_;
  cv::Mat depth_image_;

  pcl::PointCloud<pcl::PointXYZRGB> pointcloud_color_;
  pcl::PointCloud<pcl::PointXYZ> pointcloud_no_intensity_;

  sensor_msgs::CameraInfo color_camera_info_;
};

// default values
// enable / disable the output topics
constexpr bool kDefaultEnableColorImage = false;
constexpr bool kDefaultEnableColorCameraInfo = false;
constexpr bool kDefaultEnableDepthImage = false;

constexpr bool kDefaultEnablePointcloudColor = true;
constexpr bool kDefaultEnablePointcloudNoIntensity = false;

// Filter point cloud by the points color
constexpr bool kDefaultEnableColorFilter = false;
// Minimum and maximum HSV values allowed by the color filter
constexpr int kDefaultMinH = 0;
constexpr int kDefaultMinS = 0;
constexpr int kDefaultMinV = 0;
constexpr int kDefaultMaxH = 255;
constexpr int kDefaultMaxS = 255;
constexpr int kDefaultMaxV = 255;

// Rejects this number of frames between each processed one (no rejection =
// 60hz)
constexpr int kDefaultDownsampleRateFactor = 1;
// Desired number of points in the output point cloud. For values <= 0, the
// input data is not down sampled
constexpr int kDefaultTargetNumberOfSamples = -1;
// Enables a 5x5 median blur to remove noise
constexpr bool kDefaultEnableMedianFilter = true;
// Enables filter that compares the difference between the maximum and minimum
// values in a neighbourhood and rejects ones that exceed a threshold
constexpr bool kDefaultEnableMinMaxFilter = true;
// Size of minmax filter kernel
constexpr int kDefaultMinMaxFilterSize = 5;
// Min difference to reject a point
constexpr float kDefaultMinMaxFilterThreshold = 0.1;

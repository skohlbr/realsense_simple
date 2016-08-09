# realsense-simple
Unofficial ros realsense driver with some internal outlier rejection.

For most purposes you probably want the official ros driver https://github.com/intel-ros/realsense/tree/indigo-devel/realsense_camera

Disadvantages of this driver
- less features (cannot set any of realsenses internal parameters)
- less efficient
- does not output ir camera infomation or images

Advantages
- some outlier filtering and downsampling
- can be easier to get up and running with on some systems

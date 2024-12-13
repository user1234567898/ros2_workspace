cmake_minimum_required(VERSION 3.8)
project(ridar_service_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 找到ros2的基本包
# 查找ament_cmake和rosidl相关的包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)  # ROS 2的C++客户端库
find_package(rosidl_default_generators REQUIRED)  # 生成接口的工具
find_package(std_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 使用rosidl生成服务代码
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectPointCloud.srv"  # 传入所有的服务文件
  DEPENDENCIES std_msgs sensor_msgs  # 如果你的服务消息依赖于其他消息，可以在这里列出
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

#include <gtest/gtest.h>
#include <ros/ros.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "stomp_kinematics_tests");
    ros::NodeHandle nodeHandle("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    return RUN_ALL_TESTS();
}
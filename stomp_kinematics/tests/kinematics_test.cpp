#include <gtest/gtest.h>
#include <stomp_kinematics/kinematics.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

TEST(IKSolver, solve)
{
    using namespace moveit::core;

    robot_model_loader::RobotModelLoader loader("robot_description");
    const auto robot_model_ptr = loader.getModel();
    RobotState state(robot_model_ptr);
    std::string group_name = "manipulator";
    const JointModelGroup* joint_group = robot_model_ptr->getJointModelGroup(group_name);

    stomp_kinematics::kinematics::IKSolver ik_solver(state,group_name);

    Eigen::VectorXd result = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd tool_goal_tolerance = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd seed(6);
    seed << 0.655, -1.306, 1.685, 0.0, 1.07, 0;
    double ptol = 0.001;
    double rtol = 0.01;
    tool_goal_tolerance << ptol, ptol, ptol, rtol, rtol, rtol;

    Eigen::VectorXd target = seed;
    target[0] += 0.02;
    target[1] -= 0.03;
    target[2] += 0.08;
    target[3] -= 0.12;
    target[4] += 0.1;
    target[5] -= 0.02;

    state.setJointGroupPositions(joint_group,target);
    Eigen::Affine3d tool_pose = state.getFrameTransform("tool1");

    ik_solver.solve(seed,tool_pose,result,tool_goal_tolerance);

    ASSERT_NEAR(target[0], result[0], tool_goal_tolerance[0]);
    ASSERT_NEAR(target[1], result[1], tool_goal_tolerance[1]);
    ASSERT_NEAR(target[2], result[2], tool_goal_tolerance[2]);
    ASSERT_NEAR(target[3], result[3], tool_goal_tolerance[3]);
    ASSERT_NEAR(target[4], result[4], tool_goal_tolerance[4]);
    ASSERT_NEAR(target[5], result[5], tool_goal_tolerance[5]);
}

TEST(IKSolver, solve2) {
    using namespace moveit::core;

    robot_model_loader::RobotModelLoader loader("robot_description");
    const auto robot_model_ptr = loader.getModel();
    RobotState state(robot_model_ptr);
    std::string group_name = "manipulator";
    const JointModelGroup* joint_group = robot_model_ptr->getJointModelGroup(group_name);

    stomp_kinematics::kinematics::IKSolver ik_solver(state,group_name);

    Eigen::VectorXd result = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd tool_goal_tolerance = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd seed(6);
    seed << -0.718119, -1.32831, 1.29662,-0.0173133,1.46619,-1.09159;

//     After IK pose -0.940807  0.242199  0.237113   1.67248
//    0.295376   0.92897  0.223085   1.39832
//                                   -0.16624  0.279917 -0.945521   1.60873
//    0         0         0         1

    double ptol = 0.001;
    double rtol = 0.01;
    tool_goal_tolerance << ptol, ptol, ptol, rtol, rtol, rtol;

    Eigen::MatrixXd matrix(4, 4);
    matrix << -0.95138,  0.262351,  0.161391,   1.67238,
               0.287192,  0.944934,   0.15691,   1.39847,
               -0.111339,  0.195632, -0.974337,   1.60777,
                0,         0,         0,         1;



    Eigen::Affine3d tool_pose;
    tool_pose.matrix() = matrix;

    ASSERT_TRUE(ik_solver.solve(seed,tool_pose,result,tool_goal_tolerance));

    state.setJointGroupPositions(joint_group,result);
    Eigen::Affine3d fk = state.getFrameTransform("tool1");
    ROS_INFO_STREAM("fk: " << fk.matrix());
    ASSERT_TRUE(fk.matrix().isApprox(tool_pose.matrix(), 0.001));

}
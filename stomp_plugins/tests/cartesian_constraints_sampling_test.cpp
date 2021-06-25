#include <gtest/gtest.h>
#include <stomp_kinematics/kinematics.h>
#include <stomp_plugins/noise_generators/cartesian_constraints_sampling.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

class CartesianConstraintsSamplingTest : public stomp_moveit::noise_generators::CartesianConstraintsSampling
{
public:
    using stomp_moveit::noise_generators::CartesianConstraintsSampling::CartesianConstraintsSampling;
    bool exposeSetupNoiseGeneration(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                      const moveit_msgs::MotionPlanRequest &req,
                                      const stomp_core::StompConfiguration &config,
                                      moveit_msgs::MoveItErrorCodes& error_code) {
        this->setupNoiseGeneration(planning_scene, req, config, error_code);
    }

    bool exposeSetupRobotState(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                 const moveit_msgs::MotionPlanRequest &req,
                                 const stomp_core::StompConfiguration &config,
                                 moveit_msgs::MoveItErrorCodes& error_code) {
        this->setupRobotState(planning_scene, req, config, error_code);
    }

    Eigen::Affine3d exposeApplyCartesianNoise(const Eigen::VectorXd& reference_joint_pose) {
        this->applyCartesianNoise(reference_joint_pose);
    }
    bool exposeInTolerance(const Eigen::Affine3d& noisy_pose, const Eigen::Affine3d& initial_pose, double translation_tolerance, double rotation_tolerance) {
        this->translation_tolerance_ = translation_tolerance;
        this->rotation_tolerance_ = rotation_tolerance;
        this->inTolerance(noisy_pose, initial_pose);
    }
};

TEST(CartesianConstraintsSampling, inTolerance) {
    CartesianConstraintsSamplingTest generator;
    Eigen::Affine3d initial_pose = Eigen::Affine3d::Identity();
    initial_pose.translation().x() = 0.2;
    Eigen::Affine3d noisy_pose = Eigen::Affine3d::Identity() * Eigen::AngleAxisd(0.3,Eigen::Vector3d::UnitX());

    ASSERT_TRUE(generator.exposeInTolerance(noisy_pose, initial_pose, 1.0, 0.5));
    ASSERT_FALSE(generator.exposeInTolerance(noisy_pose, initial_pose, 0.1, 0.5));
    ASSERT_FALSE(generator.exposeInTolerance(noisy_pose, initial_pose, 1.0, 0.1));
    ASSERT_FALSE(generator.exposeInTolerance(noisy_pose, initial_pose, 0.1, 0.1));
}

class CartesianConstraintsSamplingFixture : public ::testing::Test
{
protected:
    robot_model_loader::RobotModelLoader loader_;
    moveit::core::RobotModelConstPtr robot_model_ptr_;
    moveit::core::RobotState state_;
    std::string group_name_;

    CartesianConstraintsSamplingFixture() : loader_("robot_description"), group_name_("manipulator"), robot_model_ptr_(loader_.getModel()), state_(robot_model_ptr_)
    {
    }

    virtual void TearDown()
    {
    }
};

TEST_F(CartesianConstraintsSamplingFixture, constructor) {
    stomp_moveit::noise_generators::CartesianConstraintsSampling generator;

    ros::NodeHandle nh;
    nh.setParam("test/rotation_tolerance", 0.2);
    nh.setParam("test/translation_tolerance", 1.0);
    std::vector<double> stddev = {0.5, 0.5, 0.5, 0.1, 0.1, 0.1};
    nh.setParam("test/stddev", stddev);

    XmlRpc::XmlRpcValue config;
    nh.getParam("test", config);

    ASSERT_TRUE(generator.initialize(robot_model_ptr_, group_name_, config));
}
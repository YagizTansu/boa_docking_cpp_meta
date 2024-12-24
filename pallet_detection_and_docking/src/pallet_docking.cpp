#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <advobot_base/Service.h>
#include <advobot_base/PalletStation.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseArray.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <actionlib_msgs/GoalID.h>
#include "pallet_detection_and_docking/utilities.h"
#include "pallet_detection_and_docking/PathCreateUtilities.h"
#include "pallet_detection_and_docking/RobotMover.h"
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>

class PalletDocking
{
public:
    explicit PalletDocking(ros::NodeHandle &nh, RobotMover &robot_mover) : nh_(nh), robot_mover_(robot_mover)
    {
        initializeSubscribersAndPublishers();
        ResetDockingPathProcess();
    }

private:
    // ROS NodeHandle and communication objects
    ros::NodeHandle &nh_;
    ros::Publisher path_publisher_;
    ros::Publisher motor_service_publisher_;
    ros::Publisher pause_publisher_;
    ros::Publisher docking_process_cancel_publisher_;
    ros::Publisher cmd_vel_publisher_;
    ros::Subscriber status_tracking_subscriber_;
    ros::Subscriber advobot_services_subscriber_;
    ros::Subscriber pallet_station_subscriber_;
    ros::Subscriber path_status_subscriber_;
    RobotMover &robot_mover_;

    // Path tracking status
    PathStatus path_status_ = PathStatus::IDLE;

    /**
     * @brief Initializes ROS subscribers and publishers.
     */
    void initializeSubscribersAndPublishers()
    {
        // Publishers

        path_publisher_ = nh_.advertise<nav_msgs::Path>("/pallet_docking_path", 1);
        motor_service_publisher_ = nh_.advertise<std_msgs::Bool>("/linear_motor_service", 1);
        pause_publisher_ = nh_.advertise<std_msgs::Bool>("/pause", 1);
        docking_process_cancel_publisher_ = nh_.advertise<actionlib_msgs::GoalID>("/follow_path/cancel", 10);
        cmd_vel_publisher_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

        // Subscribers
        advobot_services_subscriber_ = nh_.subscribe("/Services", 1, &PalletDocking::handleAdvobotServiceMessage, this);
        pallet_station_subscriber_ = nh_.subscribe("pallet_station_pose", 1, &PalletDocking::handlePalletStationPoseMessage, this);
        path_status_subscriber_ = nh_.subscribe("/follow_path/status", 10, &PalletDocking::pathStatusCallback, this);

        ROS_INFO("ROS subscribers and publishers initialized successfully.");
    }

    /**
     * @brief Generates a path to the target pallet position.
     * @param path puath for docking.
     */
    void PublishDockingPathProcess(nav_msgs::Path &path)
    {
        path_publisher_.publish(path);
        ROS_INFO("Path to pallet published.");
    }

    /**
     * @brief Resets the path tracking status to IDLE.
     */
    void ResetDockingPathProcess()
    {
        path_status_ = PathStatus::IDLE;
        ROS_INFO("Path status reset to IDLE.");
    }

    /**
     * @brief Publishes a pause command to the robot.
     * @param pause True to pause, False to resume.
     */
    void PauseDockingProcess(bool pause)
    {
        std_msgs::Bool msg;
        msg.data = pause;
        pause_publisher_.publish(msg);
        if (pause)
        {
            ROS_WARN("Pause docking pallet process!");
        }
        else
        {
            ROS_WARN("Unpause docking pallet process!");
        }
    }

    /**
     * @brief Publishes a cancel command to the robot.
     */
    void CancelDockingProcess()
    {
        // Create a GoalID message
        actionlib_msgs::GoalID msg;

        // Populate the message fields
        msg.stamp = ros::Time::now(); // Current time
        msg.id = "";

        // Publish the message
        this->docking_process_cancel_publisher_.publish(msg);

        ROS_WARN("Cancel docking pallet process!");
    }

    /**
     * @brief Waits for the path status to update to ARRIVED.
     */
    void WaitRobotFinishedToDockingProcess()
    {
        ROS_WARN("Waiting for docking to complete...");
        ros::Rate loop_rate(3); // 10 Hz
        while (path_status_ != PathStatus::ARRIVED && ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
        }
        ROS_INFO("Docking operation confirmed complete.");
    }

    /**
     * @brief Publishes a command to the linear motor.
     * @param status True to activate, False to deactivate.
     */
    void LinearMotorServicePublish(bool status)
    {
        std_msgs::Bool msg;
        msg.data = status;
        motor_service_publisher_.publish(msg);
        ROS_INFO("Linear motor: %s", status ? "Activated" : "Deactivated");
    }

    /**
     * @brief Callback for processing path status updates.
     * @param msg Goal status array message.
     */
    void pathStatusCallback(const actionlib_msgs::GoalStatusArray::ConstPtr &msg)
    {
        for (const auto &status : msg->status_list)
        {
            ROS_WARN_ONCE("Goal Status: %d", status.status);
            if (status.status == PathStatus::ARRIVED)
            {
                ROS_INFO_ONCE("Status: %d. Docking complete.", status.status);
                path_status_ = PathStatus::ARRIVED;
                for (int i = 0; i < 100; i++)
                {
                    geometry_msgs::Twist cmd_msg;
                    cmd_msg.linear.x = 0.0;  // Forward velocity
                    cmd_msg.angular.z = 0.0; // Yaw (rotation rate)
                    cmd_vel_publisher_.publish(cmd_msg);
                }
            }
        }
    }

    /**
     * @brief Handles service requests received on the /Services topic.
     * @param msg The service request message.
     */
    void handleAdvobotServiceMessage(const advobot_base::Service::ConstPtr &msg)
    {
        std::vector<std::string> services = Utilities::split(msg->service_name, ',');
        if (!services.empty() && services[0] == "service_pallet_start" && msg->request)
        {
            ROS_INFO("Received 'service_pallet_start' request. Executing pick pallet operation.");
            bool pick_status = false;
            pick_status = executePickPallet();
            if (pick_status)
            {
                executeExitPallet();
            }
        }
    }

    /**
     * @brief Processes pallet station pose messages.
     * @param msg The pallet station message.
     */
    void handlePalletStationPoseMessage(const advobot_base::PalletStation::ConstPtr &msg)
    {
        ROS_INFO("Pallet station received: x = %.2f, y = %.2f, z = %.2f",
                 msg->station_coor.pose.position.x,
                 msg->station_coor.pose.position.y,
                 msg->station_coor.pose.position.z);

        if (msg->station_type == "pick")
        {
            ROS_INFO("Station type is 'pick'. Initiating pick and exit operations.");
            bool pick_status = executePickPallet();
            if (pick_status)
            {
                executeExitPallet();
            }
        }
        else if (msg->station_type == "drop")
        {
            ROS_INFO("Station type is 'drop'. Initiating drop operation.");
            executeDropPallet();
            executeExitPallet();
        }
    }

    /**
     * @return robot_pose.
     */
    geometry_msgs::Pose getRobotPose()
    {
        auto current_pose = ros::topic::waitForMessage<geometry_msgs::Pose>("/tf_pose", nh_, ros::Duration(1));
        if (!current_pose)
        {
            throw std::runtime_error("Robot pose data unavailable.");
        }
        return *current_pose;
    }

    /**
     * @return pallet_poses.
     */
    geometry_msgs::PoseArray getPalletPoses()
    {
        auto pallet_poses = ros::topic::waitForMessage<geometry_msgs::PoseArray>("/pallet_poses", nh_, ros::Duration(1));
        if (!pallet_poses)
        {
            throw std::runtime_error("Pallet poses data unavailable.");
        }
        return *pallet_poses;
    }

    /**
     * @brief Executes pallet picking operation.
     */

    geometry_msgs::Pose calculateNewPose(const geometry_msgs::Pose &robot_pose)
    {
        // Mevcut pozisyondan x ve y alınır
        double current_x = robot_pose.position.x;
        double current_y = robot_pose.position.y;

        // Quaternion'dan yaw açısını hesapla
        tf2::Quaternion q(
            robot_pose.orientation.x,
            robot_pose.orientation.y,
            robot_pose.orientation.z,
            robot_pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // 1 metre ön ve 0.5 metre sağ hesapla
        double offset_x = 1.0 * cos(yaw) - 0.20 * sin(yaw);
        double offset_y = 1.0 * sin(yaw) + 0.20 * cos(yaw);

        // Yeni pozisyonu oluştur
        geometry_msgs::Pose new_pose;
        new_pose.position.x = current_x + offset_x;
        new_pose.position.y = current_y + offset_y;
        new_pose.position.z = robot_pose.position.z;   // Z aynı kalır
        new_pose.orientation = robot_pose.orientation; // Oryantasyon aynı

        return new_pose;
    }

    bool executePickPallet()
    {
        ROS_INFO("Starting pallet picking operation...");
        try
        {
            geometry_msgs::Pose robot_pose = getRobotPose();
            // geometry_msgs::PoseArray pallet_poses = getPalletPoses();
            // geometry_msgs::Pose nearest_pallet_pose = Utilities::getNearestPose(pallet_poses, robot_pose);
            geometry_msgs::Pose nearest_pallet_pose = calculateNewPose(robot_pose);

            ROS_INFO("Pose data successfully retrieved. Calculating path to the pallet...");
            nav_msgs::Path path = PathCreateUtilities::calculatePathCurrentPoseToTargetPose(robot_pose, nearest_pallet_pose);
            this->PublishDockingPathProcess(path);
            this->WaitRobotFinishedToDockingProcess();
            this->LinearMotorServicePublish(true); // Activate linear motor

            ROS_INFO("Finished pallet picking operation...");

            return true;
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Error during pallet picking operation: %s", e.what());
            return false;
        }
    }

    /**
     * @brief Executes pallet exit operation.
     */
    void executeExitPallet()
    {
        ROS_INFO("Starting pallet exit operation...");
        // Placeholder for implementation
        robot_mover_.robotMoveBackward(0.15, 1.3);
    }

    /**
     * @brief Executes pallet dropping operation.
     */
    void executeDropPallet()
    {
        ROS_INFO("Starting pallet dropping operation...");
        // Placeholder for implementation
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pallet_docking");
    ros::NodeHandle nh;
    RobotMover robot_mover_(nh);
    PalletDocking pallet_docking(nh, robot_mover_);
    ROS_INFO("Pallet Docking Node initialized. Waiting for operations...");
    ros::spin();
    return 0;
}
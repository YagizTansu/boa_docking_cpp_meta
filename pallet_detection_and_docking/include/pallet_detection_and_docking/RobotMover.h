#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include <cmath>
#include <ros/topic.h>

class RobotMover
{
public:
    RobotMover(ros::NodeHandle nh) : nh_(nh)
    {
        cmd_vel_publisher_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    }

    void robotMoveBackward(double speed, double target_distance)
    {
        double traveled_distance = 0.0;
        geometry_msgs::Point start_position;

        // Odometry verisini alarak başlangıç pozisyonunu kaydet
        nav_msgs::Odometry::ConstPtr odom_msg = ros::topic::waitForMessage<nav_msgs::Odometry>("/odom", ros::Duration(0.3));
        if (odom_msg)
        {
            start_position = odom_msg->pose.pose.position;
        }
        else
        {
            ROS_ERROR("Odometry message not received.");
            return;
        }

        ros::Rate rate(10); // Döngü frekansı (10 Hz)
        while (traveled_distance < target_distance && ros::ok())
        {
            // Mesafe hesapla
            nav_msgs::Odometry::ConstPtr current_odom = ros::topic::waitForMessage<nav_msgs::Odometry>("/odom", ros::Duration(2.0));
            if (current_odom)
            {
                geometry_msgs::Point current_position = current_odom->pose.pose.position;

                traveled_distance = calculateDistance(start_position, current_position);

                ROS_INFO("Traveled Distance: %.2f", traveled_distance);

                // Hedefe henüz ulaşmadıysa hareket komutu gönder
                if (traveled_distance < target_distance)
                {
                    geometry_msgs::Twist cmd_msg;
                    cmd_msg.linear.x = -std::abs(speed); // Negatif hız: Geri hareket
                    cmd_msg.angular.z = 0.0;             // Yaw (rotation rate)
                    cmd_vel_publisher_.publish(cmd_msg);
                }
            }
            else
            {
                ROS_WARN("Failed to get current odometry message.");
                break;
            }
            rate.sleep();
        }

        // Robotu durdur
        stopRobot();
        ROS_INFO("Target distance reached. Robot stopped.");
    }

private:
    ros::NodeHandle &nh_;
    ros::Publisher cmd_vel_publisher_;

    double calculateDistance(const geometry_msgs::Point &start, const geometry_msgs::Point &end)
    {
        return std::sqrt(std::pow(end.x - start.x, 2) + std::pow(end.y - start.y, 2));
    }

    void stopRobot()
    {
        for (int i = 0; i < 50; i++)
        {
            geometry_msgs::Twist cmd_msg;
            cmd_msg.linear.x = 0.0;
            cmd_msg.angular.z = 0.0;
            cmd_vel_publisher_.publish(cmd_msg);
        }
    }
};
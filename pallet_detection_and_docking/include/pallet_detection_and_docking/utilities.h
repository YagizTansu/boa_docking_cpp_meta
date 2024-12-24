#include <iostream>
#include <tuple>
#include <string>
#include <vector>
#include <sstream>
#include <cmath> // std::sin, std::cos, M_PI
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>


class Utilities
{
public:
    static std::vector<std::string> split(const std::string &str, char delimiter)
    {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    static double getDistance(const geometry_msgs::Pose &pose1, const geometry_msgs::Pose &pose2)
    {
        double dx = pose1.position.x - pose2.position.x;
        double dy = pose1.position.y - pose2.position.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // Method to get the nearest pose from a list of poses
    static geometry_msgs::Pose getNearestPose(const geometry_msgs::PoseArray &poseArray, const geometry_msgs::Pose &targetPoint)
    {
        geometry_msgs::Pose nearestPose;
        double minDistance = std::numeric_limits<double>::infinity(); // Set a very large initial value

        for (const auto &pose : poseArray.poses)
        {
            double distance = getDistance(pose, targetPoint);

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestPose = pose;
            }
        }

        return nearestPose;
    }
};
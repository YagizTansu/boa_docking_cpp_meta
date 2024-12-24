
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>

// Enum representing the different states of path tracking
enum PathStatus
{
    IDLE = 0,
    DOCKING = 1,
    ARRIVED = 3,
    FAILED = 4
};

class PathCreateUtilities
{
public:
    /**
     * @brief Converts a quaternion to Euler angles.
     * @param pose Input pose with quaternion orientation.
     * @return Roll, pitch, yaw angles as a vector.
     */
    static std::vector<double> quaternionToEuler(const geometry_msgs::Pose &pose)
    {
        tf2::Quaternion quaternion;
        quaternion.setX(pose.orientation.x);
        quaternion.setY(pose.orientation.y);
        quaternion.setZ(pose.orientation.z);
        quaternion.setW(pose.orientation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

        return {roll, pitch, yaw};
    }

    /**
     * @brief Creates a nav_msgs::Path from X and Y coordinates.
     * @param x_coords Vector of X coordinates.
     * @param y_coords Vector of Y coordinates.
     * @return The generated path.
     */
    static nav_msgs::Path createPath(const std::vector<double> &x_coords, const std::vector<double> &y_coords)
    {
        nav_msgs::Path path;
        path.header.frame_id = "map";
        path.header.stamp = ros::Time::now();

        if (x_coords.size() != y_coords.size())
        {
            ROS_ERROR("X and Y coordinates size mismatch. Path generation failed.");
            return path;
        }

        for (size_t i = 0; i < x_coords.size(); ++i)
        {
            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = "map";
            pose.header.stamp = ros::Time::now();

            pose.pose.position.x = x_coords[i];
            pose.pose.position.y = y_coords[i];
            pose.pose.position.z = 0.0;

            pose.pose.orientation.w = 1.0; // Default orientation

            path.poses.push_back(pose);
        }

        ROS_INFO("Path successfully created with %lu poses.", path.poses.size());
        return path;
    }

    static std::pair<double, double> calculateBackwardPosition(double x, double y, double yaw, double distance)
    {
        /*
        Args:
            x: Başlangıç x koordinatı
            y: Başlangıç y koordinatı
            yaw: Başlangıç yönü (radyan cinsinden)
            distance: Geri hareket edilecek mesafe

        Returns:
            std::pair<double, double>: Yeni (x, y) koordinatları
        */
        double new_x = x - distance * std::cos(yaw);
        double new_y = y - distance * std::sin(yaw);
        return std::make_pair(new_x, new_y);
    }

    /**
     * @brief Generates a path to the target pallet position.
     * @param current_pose Current robot position.
     * @param target_pose Target pallet position.
     */
    static nav_msgs::Path calculatePathCurrentPoseToTargetPose(const geometry_msgs::Pose &current_pose, const geometry_msgs::Pose &target_pose)
    {

        constexpr double step = 0.1;
        constexpr double max_distance = 1.5;
        constexpr double inside_pallet_distance = -0.8;

        std::vector<double> x_coords{current_pose.position.x};
        std::vector<double> y_coords{current_pose.position.y};
        std::vector<geometry_msgs::Quaternion> yaw_coords{current_pose.orientation};

        auto euler_angles = PathCreateUtilities::quaternionToEuler(target_pose);
        auto inside_pallet = PathCreateUtilities::calculateBackwardPosition(target_pose.position.x, target_pose.position.y, euler_angles[2], inside_pallet_distance);

        for (double dist = max_distance; dist >= step; dist -= step)
        {
            auto backward_point = PathCreateUtilities::calculateBackwardPosition(inside_pallet.first, inside_pallet.second, euler_angles[2], dist);
            x_coords.push_back(backward_point.first);
            y_coords.push_back(backward_point.second);
            yaw_coords.push_back(current_pose.orientation);
        }

        x_coords.push_back(inside_pallet.first);
        y_coords.push_back(inside_pallet.second);
        yaw_coords.push_back(target_pose.orientation);

        nav_msgs::Path path = PathCreateUtilities::createCubicSplinePath(x_coords, y_coords, yaw_coords, 15);

        ROS_INFO("Path to pallet generated.");

        return path;
    }

    // Function to calculate cubic spline coefficients
    static std::vector<Eigen::Vector4d> calculateSplineCoefficients(const std::vector<double> &x, const std::vector<double> &y)
    {
        int n = x.size();
        std::vector<Eigen::Vector4d> coefficients(n - 1);

        // Create matrices for spline calculations
        Eigen::MatrixXd A(n - 2, n);
        A.setZero();

        for (int i = 1; i < n - 1; ++i)
        {
            A(i - 1, i - 1) = 2 * (x[i] - x[i - 1]);
            A(i - 1, i) = 1 * (x[i + 1] - x[i - 1]);
        }

        Eigen::MatrixXd b(n - 2, 1);
        for (int i = 1; i < n - 1; ++i)
        {
            b(i - 1) = 3 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]));
        }

        // Solve the system of equations
        Eigen::VectorXd c = A.colPivHouseholderQr().solve(b);

        // Calculate coefficients
        for (int i = 0; i < n - 1; ++i)
        {
            coefficients[i](0) = y[i];
            coefficients[i](1) = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (x[i + 1] - x[i]) * (c(i) + c(i + 1)) / 3.0;
            coefficients[i](2) = c(i);
            coefficients[i](3) = (c(i + 1) - c(i)) / (3.0 * (x[i + 1] - x[i]));
        }

        return coefficients;
    }

    // Function to evaluate spline at a given point
    static double evaluateSpline(const std::vector<Eigen::Vector4d> &coefficients, double t, double x)
    {
        int segment = 0;
        while (segment < coefficients.size() - 1 && x > coefficients[segment + 1](0))
        {
            segment++;
        }

        double dx = x - coefficients[segment](0);
        return coefficients[segment](0) + coefficients[segment](1) * dx + coefficients[segment](2) * dx * dx + coefficients[segment](3) * dx * dx * dx;
    }

    static nav_msgs::Path createCubicSplinePath(const std::vector<double> &x_coords, const std::vector<double> &y_coords, std::vector<geometry_msgs::Quaternion>& yaw_coords, size_t interpolation_points = 100)
    {
        nav_msgs::Path path;
        path.header.frame_id = "map";
        path.header.stamp = ros::Time::now();

        if (x_coords.size() != y_coords.size())
        {
            ROS_ERROR("X and Y coordinates size mismatch. Path generation failed.");
            return path;
        }

        if (x_coords.size() < 2)
        {
            ROS_ERROR("Not enough points to create a cubic spline. Path generation failed.");
            return path;
        }

        // Convert input coordinates to Eigen vectors
        Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x_coords.data(), x_coords.size());
        Eigen::VectorXd y_eigen = Eigen::Map<const Eigen::VectorXd>(y_coords.data(), y_coords.size());

        // Parameterize points with cumulative distance (for consistent spacing)
        Eigen::VectorXd t(x_coords.size());
        t(0) = 0.0;
        for (size_t i = 1; i < x_coords.size(); ++i)
        {
            t(i) = t(i - 1) + std::hypot(x_coords[i] - x_coords[i - 1], y_coords[i] - y_coords[i - 1]);
        }

        // Create cubic splines for x and y coordinates
        Eigen::Spline<double, 1> spline_x = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(x_eigen.transpose(), 3, t);
        Eigen::Spline<double, 1> spline_y = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(y_eigen.transpose(), 3, t);

        // Interpolate along the spline
        double t_min = t(0);
        double t_max = t(t.size() - 1);
        for (size_t i = 0; i < interpolation_points; ++i)
        {
            double t_interp = t_min + (t_max - t_min) * i / (interpolation_points - 1);

            Eigen::Matrix<double, 1, 1> x_val = spline_x(t_interp);
            Eigen::Matrix<double, 1, 1> y_val = spline_y(t_interp);

            geometry_msgs::PoseStamped pose;
            pose.header.frame_id = "map";
            pose.header.stamp = ros::Time::now();

            pose.pose.position.x = x_val(0);
            pose.pose.position.y = y_val(0);
            pose.pose.position.z = 0.0;

            pose.pose.orientation = yaw_coords[0]; // Default orientation

            path.poses.push_back(pose);
        }

        ROS_INFO("Cubic spline path successfully created with %lu poses.", path.poses.size());
        return path;
    }
};

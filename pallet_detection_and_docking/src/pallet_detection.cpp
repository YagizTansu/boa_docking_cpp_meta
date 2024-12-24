#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>
#include <pcl_ros/surface/convex_hull.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <cmath>

#include <advobot_base/PalletStation.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <vector>

struct Quaternion
{
    double w, x, y, z;
};

class LidarDBSCANClustering
{
public:
    LidarDBSCANClustering(ros::NodeHandle &nh) : tolerance(0.25), min_cluster_size(400), max_cluster_size(2000), tf_listener_(tf_buffer_)
    {
        laser_sub = nh.subscribe("/scan_fork", 1, &LidarDBSCANClustering::scanCallback, this);
        pallet_station_subscriber = nh.subscribe("pallet_station_pose", 1, &LidarDBSCANClustering::handlePalletStationPoseMessage, this);

        marker_pub = nh.advertise<visualization_msgs::Marker>("convex_hull_marker", 1);
        lidar_filtered_area_publisher = nh.advertise<visualization_msgs::Marker>("filtered_lidar_area_marker", 1);

        // Define static polygon points in the map frame
        polygon_points_ = {
            createPoint(6.6, -0.5, 0.0),
            createPoint(6.6, -1.5, 0.0),
            createPoint(5.4, -1.5, 0.0),
            createPoint(5.4, -0.5, 0.0)};
        ROS_INFO("LIDAR DBSCAN Clustering Node Started...");
    }

private:
    ros::Subscriber laser_sub;
    ros::Subscriber pallet_station_subscriber;

    ros::Publisher marker_pub;
    ros::Publisher lidar_filtered_area_publisher;

    float tolerance;
    int min_cluster_size;
    int max_cluster_size;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::vector<geometry_msgs::Point> polygon_points_;

    void handlePalletStationPoseMessage(const advobot_base::PalletStation::ConstPtr &msg)
    {
        // Quaternion'dan yön bilgisi türetmek için dönüşüm
        tf::Quaternion quat;
        tf::quaternionMsgToTF(msg->station_coor.pose.orientation, quat);
        double roll, pitch, yaw;
        tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

        // Yön vektörlerini oluştur
        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);

        // Dikdörtgen yarıçapı
        double half_width = 0.5;
        double half_height = 0.5;

        // Dikdörtgen köşeleri
        geometry_msgs::Point p1, p2, p3, p4;

        // Köşe 1
        p1.x = msg->station_coor.pose.position.x + (half_width * cos_yaw - half_height * sin_yaw);
        p1.y = msg->station_coor.pose.position.y + (half_width * sin_yaw + half_height * cos_yaw);
        p1.z = 0.0;

        // Köşe 2
        p2.x = msg->station_coor.pose.position.x + (half_width * cos_yaw + half_height * sin_yaw);
        p2.y = msg->station_coor.pose.position.y - (half_width * sin_yaw - half_height * cos_yaw);
        p2.z = 0.0;

        // Köşe 3
        p3.x = msg->station_coor.pose.position.x - (half_width * cos_yaw + half_height * sin_yaw);
        p3.y = msg->station_coor.pose.position.y - (half_width * sin_yaw + half_height * cos_yaw);
        p3.z = 0.0;

        // Köşe 4
        p4.x = msg->station_coor.pose.position.x - (half_width * cos_yaw - half_height * sin_yaw);
        p4.y = msg->station_coor.pose.position.y + (half_width * sin_yaw - half_height * cos_yaw);
        p4.z = 0.0;

    }

    geometry_msgs::Point createPoint(double x, double y, double z)
    {
        geometry_msgs::Point point;
        point.x = x;
        point.y = y;
        point.z = z;
        return point;
    }

    bool isPointInPolygon(const geometry_msgs::Point &point, const std::vector<geometry_msgs::Point> &polygon)
    {
        int n = polygon.size();
        int crossings = 0;

        for (int i = 0; i < n; ++i)
        {
            const geometry_msgs::Point &p1 = polygon[i];
            const geometry_msgs::Point &p2 = polygon[(i + 1) % n];

            if (((p1.y > point.y) != (p2.y > point.y)) &&
                (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x))
            {
                crossings++;
            }
        }

        return (crossings % 2 == 1);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
    {
        publishRectangle(polygon_points_[0], polygon_points_[1], polygon_points_[2], polygon_points_[3]);
        try
        {
            // Get the transform from lidar_fork to map
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform("map", "lidar_fork", ros::Time(0));

            // Prepare a new LaserScan message for filtered data
            auto filtered_scan = boost::make_shared<sensor_msgs::LaserScan>(*scan_msg);

            filtered_scan->ranges.clear();

            // Iterate through LaserScan ranges
            for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
            {
                double angle = scan_msg->angle_min + i * scan_msg->angle_increment;
                double range = scan_msg->ranges[i];

                // Skip invalid range values
                if (range < scan_msg->range_min || range > scan_msg->range_max)
                {
                    filtered_scan->ranges.push_back(std::numeric_limits<float>::infinity());
                    continue;
                }

                // Calculate point in lidar_fork frame
                geometry_msgs::PointStamped point_in_lidar;
                point_in_lidar.header.frame_id = "lidar_fork";
                point_in_lidar.point.x = range * cos(angle);
                point_in_lidar.point.y = range * sin(angle);
                point_in_lidar.point.z = 0.0;

                // Transform point to map frame
                geometry_msgs::PointStamped point_in_map;
                tf2::doTransform(point_in_lidar, point_in_map, transform);

                // Check if the point is within the polygon
                if (isPointInPolygon(point_in_map.point, polygon_points_))
                {
                    filtered_scan->ranges.push_back(range);
                }
                else
                {
                    filtered_scan->ranges.push_back(std::numeric_limits<float>::infinity());
                }
            }

            auto cloud = convertLaserScanToPointCloud(filtered_scan);
            auto possible_pallet_indicies = performDBSCAN(cloud);
            auto pallet_cloud = createClusteredPointCloud(cloud, possible_pallet_indicies);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("Transform failed: %s", ex.what());
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr convertLaserScanToPointCloud(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (size_t i = 0; i < scan_msg->ranges.size(); ++i)
        {
            if (std::isfinite(scan_msg->ranges[i]))
            {
                float angle = scan_msg->angle_min + i * scan_msg->angle_increment;
                pcl::PointXYZ point;
                point.x = scan_msg->ranges[i] * cos(angle);
                point.y = scan_msg->ranges[i] * sin(angle);
                point.z = 0.0;
                cloud->points.push_back(point);
            }
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        return cloud;
    }

    std::vector<pcl::PointIndices> performDBSCAN(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(tolerance);
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        ec.extract(cluster_indices);
        int cluster_id = 0;
        for (const auto &indices : cluster_indices)
        {
            publishConvexHull(cloud, indices, 10 + cluster_id++);
        }

        return cluster_indices;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr createClusteredPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::vector<pcl::PointIndices> &cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &indices : cluster_indices)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud2(new pcl::PointCloud<pcl::PointXYZ>);

            for (const auto &idx : indices.indices)
            {
                pcl::PointXYZ point;
                point.x = cloud->points[idx].x;
                point.y = cloud->points[idx].y;
                point.z = cloud->points[idx].z;
                clustered_cloud->points.push_back(point);
                clustered_cloud2->points.push_back(point);
            }
        }

        clustered_cloud->width = clustered_cloud->points.size();
        clustered_cloud->height = 1;
        clustered_cloud->is_dense = true;

        return clustered_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr extractClusterPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices &cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &idx : cluster_indices.indices)
        {
            cluster_points->points.push_back(cloud->points[idx]);
        }
        return cluster_points;
    }

    pcl::PointCloud<pcl::PointXYZ> computeConvexHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster_points)
    {
        pcl::ConvexHull<pcl::PointXYZ> chull;
        pcl::PointCloud<pcl::PointXYZ> hull_cloud;
        chull.setInputCloud(cluster_points);
        chull.reconstruct(hull_cloud);
        return hull_cloud;
    }

    void computeMinAreaRectangle(const pcl::PointCloud<pcl::PointXYZ> &hull_cloud, cv::Point2f rect_points[4])
    {
        std::vector<cv::Point2f> points;
        for (const auto &point : hull_cloud.points)
        {
            points.emplace_back(point.x, point.y);
        }

        cv::RotatedRect minRect = cv::minAreaRect(points);
        minRect.points(rect_points);
    }

    double calculateYaw(const cv::Point2f &p1, const cv::Point2f &p2)
    {
        // Calculate the differences in the x and y coordinates
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;

        // Use atan2 to compute the angle in radians
        double yaw = std::atan2(dy, dx) + (M_PI / 2);
        if (yaw < (M_PI / 2))
        {
            yaw += M_PI;
        }
        return yaw; // Return the angle in radians
    }

    Quaternion eulerToQuaternion(double roll, double pitch, double yaw)
    {
        // Calculate half angles
        double cy = cos(yaw * 0.5);
        double sy = sin(yaw * 0.5);
        double cp = cos(pitch * 0.5);
        double sp = sin(pitch * 0.5);
        double cr = cos(roll * 0.5);
        double sr = sin(roll * 0.5);

        Quaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;

        return q;
    }

    std::vector<geometry_msgs::Pose> computeRectangleCenters(const cv::Point2f rect_points[4])
    {
        std::vector<geometry_msgs::Pose> center_points;

        // Ana merkez noktası (Uzun kenarın orta noktası)
        geometry_msgs::Pose center_point{};
        center_point.position.x = (rect_points[3].x + rect_points[0].x) / 2;
        center_point.position.y = (rect_points[3].y + rect_points[0].y) / 2;
        center_point.position.z = 0.0;
        double yaw = calculateYaw(rect_points[0], rect_points[3]);
        Quaternion q = eulerToQuaternion(0, 0, yaw);
        center_point.orientation.x = q.x;
        center_point.orientation.y = q.y;
        center_point.orientation.z = q.z;
        center_point.orientation.w = yaw;
        center_points.push_back(center_point);

        // Diğer kenarların orta noktalarını hesapla ve mesafeye göre filtrele
        for (size_t i = 0; i < 3; i++)
        {
            geometry_msgs::Pose center_point{};
            double distance = std::sqrt(std::pow(rect_points[i + 1].x - rect_points[i].x, 2) + std::pow(rect_points[i + 1].y - rect_points[i].y, 2));
            double yaw = calculateYaw(rect_points[i], rect_points[i + 1]);
            Quaternion q = eulerToQuaternion(0, 0, yaw);
            center_point.position.x = (rect_points[i].x + rect_points[i + 1].x) / 2;
            center_point.position.y = (rect_points[i].y + rect_points[i + 1].y) / 2;
            center_point.position.z = 0.0;
            center_point.orientation.x = q.x;
            center_point.orientation.y = q.y;
            center_point.orientation.z = q.z;
            center_point.orientation.w = yaw;

            if (distance > 0.6 && distance < 1.0)
            {
                center_points.push_back(center_point);
            }
        }

        return center_points;
    }

    void printPose(const geometry_msgs::Pose &pose)
    {
        // Print position
        std::cout << "###########################" << std::endl;
        std::cout << "Position:" << std::endl;
        std::cout << "  x: " << pose.position.x << std::endl;
        std::cout << "  y: " << pose.position.y << std::endl;
        std::cout << "  z: " << pose.position.z << std::endl;

        // Print orientation
        std::cout << "Orientation:" << std::endl;
        std::cout << "  x: " << pose.orientation.x << std::endl;
        std::cout << "  y: " << pose.orientation.y << std::endl;
        std::cout << "  z: " << pose.orientation.z << std::endl;
        std::cout << "  w: " << pose.orientation.w << std::endl;
        std::cout << "###########################" << std::endl;
    }

    geometry_msgs::Pose findClosestPoint(const std::vector<geometry_msgs::Pose> &center_points)
    {
        geometry_msgs::Pose closest_pose;
        double min_distance = std::numeric_limits<double>::infinity();

        for (const auto &pose : center_points)
        {
            double distance = std::sqrt(std::pow(pose.position.x, 2) + std::pow(pose.position.y, 2));
            if (distance < min_distance)
            {
                min_distance = distance;
                closest_pose = pose;
            }
        }
        // printPose(closest_pose);
        return closest_pose;
    }

    void publishConvexHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices &cluster_indices, int id)
    {
        // 1. Küme noktalarını ayıkla
        auto cluster_points = extractClusterPoints(cloud, cluster_indices);

        // 2. Dışbükey kova hesapla
        auto hull_cloud = computeConvexHull(cluster_points);

        // 3. Minimum alanlı dönen dikdörtgeni hesapla
        cv::Point2f rect_points[4];
        computeMinAreaRectangle(hull_cloud, rect_points);

        // 4. Dikdörtgenin merkezlerini hesapla
        auto center_points = computeRectangleCenters(rect_points);

        // 5. En yakın noktayı bul
        auto closest_pose = findClosestPoint(center_points);

        // 6. Sonuçları yayınla
        publishPalletFront(closest_pose, id);
        publishRotatedRectangle(rect_points, id);
    }

    void publishPalletFront(geometry_msgs::Pose start_pose, int id)
    {
        visualization_msgs::Marker arrow_marker;
        arrow_marker.header.frame_id = "lidar_fork";
        arrow_marker.header.stamp = ros::Time::now();
        arrow_marker.ns = "arrow_marker";
        arrow_marker.id = id * 10;
        arrow_marker.type = visualization_msgs::Marker::ARROW;
        arrow_marker.action = visualization_msgs::Marker::ADD;
        arrow_marker.pose.orientation.w = 1.0;

        // Define arrow scale: shaft diameter and head diameter/length
        arrow_marker.scale.x = 0.05; // Shaft diameter
        arrow_marker.scale.y = 0.1;  // Head diameter
        arrow_marker.scale.z = 0.15; // Head length

        // Define arrow color
        arrow_marker.color.r = 0.0;
        arrow_marker.color.g = 1.0;
        arrow_marker.color.b = 1.0;
        arrow_marker.color.a = 1.0;

        // Define the start and end points of the arrow
        geometry_msgs::Point start_point;
        start_point.x = start_pose.position.x;
        start_point.y = start_pose.position.y;
        start_point.z = 0.0;

        geometry_msgs::Point end_point;
        tf2::Quaternion quat(start_pose.orientation.x, start_pose.orientation.y,
                             start_pose.orientation.z, start_pose.orientation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

        // Yön (yaw) doğrultusunda 0.5 metre ilerlet
        end_point.x = start_point.x + 0.5 * cos(start_pose.orientation.w);
        end_point.y = start_point.y + 0.5 * sin(start_pose.orientation.w);
        end_point.z = 0.0;

        // Add the points to the marker
        arrow_marker.points.push_back(start_point);
        arrow_marker.points.push_back(end_point);

        // Publish the arrow marker
        marker_pub.publish(arrow_marker);
    }

    void publishRotatedRectangle(const cv::Point2f *rect_points, int id)
    {
        visualization_msgs::Marker rect_marker;
        rect_marker.header.frame_id = "lidar_fork";
        rect_marker.header.stamp = ros::Time::now();
        rect_marker.ns = "min_rotated_rectangle";
        rect_marker.id = id;
        rect_marker.type = visualization_msgs::Marker::LINE_STRIP;
        rect_marker.action = visualization_msgs::Marker::ADD;
        rect_marker.pose.orientation.w = 1.0;
        rect_marker.scale.x = 0.02;
        rect_marker.color.r = 0.0;
        rect_marker.color.g = 1.0;
        rect_marker.color.b = 1.0;
        rect_marker.color.a = 1.0;

        for (int i = 0; i < 4; i++)
        {
            geometry_msgs::Point p;
            p.x = rect_points[i].x;
            p.y = rect_points[i].y;
            p.z = 0.0;
            rect_marker.points.push_back(p);
        }
        rect_marker.points.push_back(rect_marker.points.front());

        marker_pub.publish(rect_marker);
    }

    void publishRectangle(const geometry_msgs::Point &p1,
                          const geometry_msgs::Point &p2,
                          const geometry_msgs::Point &p3,
                          const geometry_msgs::Point &p4)
    {
        // Marker mesajını oluştur
        visualization_msgs::Marker rectangle_marker;
        rectangle_marker.header.frame_id = "map"; // Çerçeve adı
        rectangle_marker.header.stamp = ros::Time::now();
        rectangle_marker.ns = "rectangle";
        rectangle_marker.id = 0;
        rectangle_marker.type = visualization_msgs::Marker::LINE_STRIP; // Çizgi tipi
        rectangle_marker.action = visualization_msgs::Marker::ADD;

        // Çizgi rengi ve boyutu
        rectangle_marker.scale.x = 0.05; // Çizgi kalınlığı
        rectangle_marker.color.r = 0.0;
        rectangle_marker.color.g = 1.0;
        rectangle_marker.color.b = 0.0;
        rectangle_marker.color.a = 1.0; // Saydamlık

        // Çizgi noktalarını ekle
        rectangle_marker.points.push_back(p1);
        rectangle_marker.points.push_back(p2);
        rectangle_marker.points.push_back(p3);
        rectangle_marker.points.push_back(p4);
        rectangle_marker.points.push_back(p1); // Dikdörtgeni kapatmak için ilk noktaya geri dön

        // Marker'ı yayınla
        lidar_filtered_area_publisher.publish(rectangle_marker);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_dbscan_clustering");
    ros::NodeHandle nh;

    LidarDBSCANClustering dbscan_clustering(nh);

    ros::spin();

    return 0;
}
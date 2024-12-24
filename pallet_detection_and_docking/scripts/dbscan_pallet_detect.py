#!/usr/bin/env python3
import tf 
import math
import rospy
import numpy as np
from advobot_base.msg import Service
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point,PoseStamped,Pose,PoseArray
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

class IsPallet(Enum):
    YES = 1
    COMBINED = 2
    NO = 3

class PalletDetector:

    def __init__(self):
        self.pallet_front_points = rospy.Publisher("/pallet_front_points", MarkerArray, queue_size=1)
        self.publisher = rospy.Publisher("/estimated_pallet_clusters", MarkerArray, queue_size=1)
        self.bbox_publisher = rospy.Publisher("/pallet_bounding_boxes", MarkerArray, queue_size=1) # Publish the bounding boxes of pallet
        self.sub_services = rospy.Subscriber('/services', Service, self.callback_service)
        self.pose_array_publisher = rospy.Publisher('/pallet_poses', PoseArray, queue_size=1) # Publish the pallets pose

        # DBSCAN parameters
        self.eps = 0.50            # maximum distance between two points to be considered neighbors
        self.min_samples = 13     # minimum number of points to form a cluster

        # Pallet parameters
        self.pallet_length = 1.2  # meters
        self.pallet_width = 0.8   # meters
        self.tolerance = 0.13     # meters, tolerance for dimension variation

        # Transformation parameters
        self.wait_transform_timeout = 2.0
        self.max_dist_to_middle_point = 2.0

        self.start_action_check=False

    def callback_service(self, data):
        incoming_service_name = data.service_name.split(",")

        if incoming_service_name[0] == "service_search_pallet" and data.request == True:
            self.start_action_check=True

        incoming_service_name = ""
        data.request = False
               
    def find_pallet_corners(self, cluster):
        hull = ConvexHull(cluster) # Find the convex hull of the cluster
        boundary_points = cluster[hull.vertices]
        middles = []
        threshold_distance = 0.1  # Define a threshold distance

        # Find the nearest middle_point
        min_distance = float('inf')
        pallet_middle_point = None
        yaw = None

        for edge in hull.simplices:
            point1 = cluster[edge[0]]
            point2 = cluster[edge[1]]

            middle_point = (point1 + point2) / 2
            # Check if middle_point is far enough from all boundary points
            if not any(np.linalg.norm(middle_point - b_point) < threshold_distance for b_point in boundary_points):
                distance = np.linalg.norm([0.0 , 0.0] - middle_point)
                middles.append(middle_point)
                # print(distance)
                if distance < min_distance:
                    min_distance = distance
                    pallet_middle_point = middle_point

                    if point2[0] - point1[0] != 0:
                        yaw = math.atan((point2[1] - point1[1]) / (point2[0] - point1[0])) + math.pi / 2
                    else:
                        yaw = (np.pi / 2 if point2[1] > point1[1] else -np.pi / 2)+math.pi/2 # Vertical line case

  
        if(yaw !=None and yaw <= 1.57): 
            yaw += math.pi 

        return middles, pallet_middle_point, yaw
        
    def process_lidar_data(self, lidar_data):
        angle_increment = (2 * np.pi) / len(lidar_data)
        data_points = []
        for i, distance in enumerate(lidar_data):
            angle = i * angle_increment
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            if distance < self.max_dist_to_middle_point:
                data_points.append([x, y])
        
        clusters = []
        if len(data_points) == 0:
            clusters = []
        else:
            data_points = np.array(data_points)
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(data_points)
            clusters = self.group_clusters(data_points, labels)
        
        return clusters

    def group_clusters(self, data_points, labels):
        unique_labels = set(labels)
        clusters = [data_points[labels == label] for label in unique_labels if label != -1]
        return clusters
    
    def find_pallets(self, clusters):
        pallets = []
        for cluster in clusters:
            if self.is_pallet(cluster) == IsPallet.YES:
                pallets.append(cluster)
            elif self.is_pallet(cluster) == IsPallet.COMBINED:

                kmeans = KMeans(n_clusters=2, random_state=42)
                kmeans.fit(cluster)
                labels = kmeans.labels_

                if self.is_pallet(cluster[labels == 0]) == IsPallet.YES:
                    pallets.append(cluster[labels == 0])

                if self.is_pallet(cluster[labels == 1]) == IsPallet.YES:
                    pallets.append(cluster[labels == 1])
        return pallets
    
    def is_pallet(self, cluster):
        x_coords = [point[0] for point in cluster]
        y_coords = [point[1] for point in cluster]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        length = max_x - min_x
        width = max_y - min_y

        # hull = ConvexHull(cluster)

        # # Get the boundary points of the convex hull
        # hull_vertices = cluster[hull.vertices]

        # # Find the width and length of the convex hull
        # length = np.max(hull_vertices[:, 0]) - np.min(hull_vertices[:, 0])
        # width = np.max(hull_vertices[:, 1]) - np.min(hull_vertices[:, 1])

        if (length >= 0.1 and length <= 1.3 and  self.pallet_width - self.tolerance <= width <= self.pallet_width + self.tolerance):
            return IsPallet.YES
        elif (width >= 1.4 and width <= 2.1 and length >= 0.1 and length <= 1.3 ):
            return IsPallet.COMBINED
        else:
            return IsPallet.NO

    def publish_clusters(self, clusters):
        marker_array = MarkerArray()
        colors = [  (1.0, 0.0, 0.0, 1.0),  # Red
                    (0.0, 1.0, 0.0, 1.0),  # Green
                    (0.0, 0.0, 1.0, 1.0),  # Blue
                    (1.0, 1.0, 0.0, 1.0),  # Yellow
                    (1.0, 0.0, 1.0, 1.0),  # Magenta
                    (0.0, 1.0, 1.0, 1.0),  # Cyan
                    (0.5, 0.5, 0.5, 1.0),  # Gray
                    (1.0, 0.5, 0.0, 1.0),  # Orange
                    (0.5, 0.0, 1.0, 1.0),  # Purple
                    (0.0, 0.5, 1.0, 1.0),  # Light Blue
                    (0.5, 1.0, 0.0, 1.0),  # Light Green
                    (0.0, 1.0, 0.5, 1.0),  # Dark Cyan
                    (1.0, 0.0, 0.5, 1.0),  # Dark Magenta
                    (0.0, 0.5, 0.5, 1.0),  # Dark Green
                    (0.5, 0.0, 0.5, 1.0),  # Dark Purple
                    (0.5, 0.5, 0.0, 1.0)]  # Dark Yellow
        
        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = "base_link"  # Replace with your LiDAR frame ID
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1 # Size of the points
            marker.scale.y = 0.1

            color = colors[i % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]  # Alpha (opacity)
            

            # Adding points to the marker
            for point in cluster:
                p = Point() 
                p.x = point[0]*-1
                p.y = point[1]*-1
                p.z = 0  # Assuming 2D LiDAR

                marker.points.append(p)

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)

    def publish_pallet_front_points(self,corners):
        marker_array = MarkerArray()
        color = [0.1, 0.0, 0.0, 1.0]  # Cyan
        for i, corner in enumerate(corners):        
            marker = Marker()
            marker.header.frame_id = "base_link"  # Replace with your LiDAR frame ID
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05 # Size of the points
            marker.scale.y = 0.05

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]  # Alpha (opacity)

            p = Point() 
            p.x = -corner[0]
            p.y = -corner[1]
            p.z = 0  # Assuming 2D LiDAR   

            marker.points.append(p)

            marker_array.markers.append(marker)

        self.pallet_front_points.publish(marker_array)
    
    def publish_bounding_boxes(self, pallet,i,corners):
        marker_array = MarkerArray()
        marker = self.create_bbox_marker( i,corners)
        marker_array.markers.append(marker)
        self.bbox_publisher.publish(marker_array)

    def create_bbox_marker(self,marker_id,corners):
        marker = Marker()
        marker.header.frame_id = "base_link"  # Replace with your LiDAR frame ID
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # Width of the lines

        # Set the color of the bounding box
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha

        # Define the corners of the bounding box
        marker.points = []

        for corner in corners:
            p = Point()
            p.x = corner[0]*-1
            p.y = corner[1]*-1
            p.z = 0
            marker.points.append(p)

        return marker
    
    def transformPoseBaseScanToMap(self,corner,quaternion):
        try:
            listener = tf.TransformListener()
            listener.waitForTransform("/base_link", "/map",  rospy.Time(0),rospy.Duration(self.wait_transform_timeout))

            point=PoseStamped()
            point.header.frame_id = "base_link"
            point.header.stamp = rospy.Time(0)
            point.pose.position.x= corner[0]
            point.pose.position.y= corner[1]
            point.pose.position.z=0.0
            point.pose.orientation.x = quaternion[0]
            point.pose.orientation.y = quaternion[1]
            point.pose.orientation.z = quaternion[2]
            point.pose.orientation.w = quaternion[3]

            return listener.transformPose("map",point)
        
        except tf.Exception as e:
            rospy.logwarn("Transformation failed: {}".format(str(e)))
            # Handle the exception as needed, e.g., return a default PoseStamped or raise an error.
            return None  # You might want to modify this line based on your requirements
    
    def publish_pallets_pose(self,pallets_pose):
        pose_array_msg = PoseArray()

        # Set the header of the PoseArray message
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = 'base_link'  # Adjust the frame_id as needed
        pose_array_msg.poses = pallets_pose

        self.pose_array_publisher.publish(pose_array_msg)
            
    def operate(self):
        laser = rospy.wait_for_message("/scan", LaserScan, 1.0) # Wait for a LaserScan message
        objects = self.process_lidar_data(list(laser.ranges)) # Process the LiDAR data to find clusters
        pallets = self.find_pallets(objects) # Find pallets in the clusters
        pallet_poses = []
        print(objects)
        for i,pallet in enumerate(pallets):
            corners ,midle_point , pallet_yaw = self.find_pallet_corners(pallet)
            if pallet_yaw != None:

                # Convert point base_link to map
                quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, pallet_yaw)
                midle_point = [-midle_point[0],-midle_point[1]]

                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "base_link"  # Adjust the frame_id as needed
                pose_msg.pose.position.x = midle_point[0]
                pose_msg.pose.position.y = midle_point[1]
                pose_msg.pose.position.z = 0.0
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]

                if(pose_msg != None):
                    pallet = Pose()
                    pallet.position = pose_msg.pose.position
                    pallet.orientation = pose_msg.pose.orientation
                    pallet_poses.append(pallet)
                    self.publish_pallet_front_points(corners) # Publish the pallet front points
                    self.publish_bounding_boxes(pallet,i,corners)
                else:
                    rospy.logwarn("Pallet pose not published because of transformation error")

        self.publish_clusters(pallets) # Publish the pallet clusters
        # rospy.loginfo("Pallets found: {}".format(len(pallet_poses)))
        if len(pallet_poses) > 0:
            self.publish_pallets_pose(pallet_poses) # Publish the pallets pose

def main():
    rospy.init_node('pallet_detector_node')
    detector = PalletDetector()
    rate = rospy.Rate(10.0) # Rate at which to publish 10 Hz
    while not rospy.is_shutdown():
        detector.operate()
        # if detector.start_action_check == True:
        #     detector.operate()
        #     detector.start_action_check = False
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

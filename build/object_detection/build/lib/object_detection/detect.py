import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# Simple Centroid Tracker
# -------------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid tuple
        self.max_distance = max_distance

    def update(self, rects):
        """
        rects: list of bounding boxes, each defined as (x_min, y_min, x_max, y_max)
        """
        input_centroids = []
        for (x_min, y_min, x_max, y_max) in rects:
            cx = int((x_min + x_max) / 2.0)
            cy = int((y_min + y_max) / 2.0)
            input_centroids.append((cx, cy))

        # If no objects currently tracked, register all input centroids.
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.objects[self.next_object_id] = centroid
                self.next_object_id += 1
            return self.objects

        # Compute distance matrix between existing object centroids and new centroids.
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (cx, cy) in enumerate(input_centroids):
                D[i, j] = np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2)

        # Find the smallest distance for each existing object.
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            used_rows.add(row)
            used_cols.add(col)

        # Register new objects for unmatched new centroids.
        for j in range(len(input_centroids)):
            if j not in used_cols:
                self.objects[self.next_object_id] = input_centroids[j]
                self.next_object_id += 1

        return self.objects

# -------------------------
# Detection and Tracking Node
# -------------------------
class ObjectDetectionTrackingNode(Node):
    def __init__(self):
        super().__init__('object_detection_tracking_node')
        self.bridge = CvBridge()
        # Load YOLOv8 model (will auto-download 'yolov8n.pt' if not present)
        self.model = YOLO("yolov8n.pt")
        
        # Subscribe to the camera feed topic (update if necessary)
        self.image_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for tracking results (text summary)
        self.tracking_pub = self.create_publisher(String, '/tracked_objects', 10)
        
        # Publisher for output image (with detections and tracking overlay)
        self.image_pub = self.create_publisher(Image, '/object_detection/tracked_image', 10)
        
        # Initialize the centroid tracker
        self.tracker = CentroidTracker(max_distance=50)
    
    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Run YOLO object detection
        results = self.model(frame)
        # Ensure we work with a single result
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        rects = []  # To hold bounding box coordinates
        if result.boxes is not None:
            # Get bounding boxes, confidences, and class indices as numpy arrays.
            boxes_np = result.boxes.xyxy.cpu().numpy()  # shape (n,4)
            confidences = result.boxes.conf.cpu().numpy()  # shape (n,)
            classes = result.boxes.cls.cpu().numpy()       # shape (n,)
            
            for i, box in enumerate(boxes_np):
                x_min, y_min, x_max, y_max = box
                rects.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                label = result.names[int(classes[i])]
                conf = confidences[i]
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update tracker with current detections and get tracked objects (ID -> centroid)
        tracked_objects = self.tracker.update(rects)
        
        # Prepare a string with tracking information
        tracking_info = []
        for object_id, centroid in tracked_objects.items():
            tracking_info.append(f"ID {object_id}: {centroid}")
            # Draw the centroid and object ID on the frame
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        self.tracking_pub.publish(String(data=", ".join(tracking_info)))
        
        # Publish the processed image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(out_msg)
        
        # Optionally, display the frame for debugging
        cv2.imshow("Object Detection and Tracking", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


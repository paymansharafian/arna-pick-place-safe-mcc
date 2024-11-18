import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from queue import Queue
from threading import Thread
from typing import Optional, Tuple, List, Callable
from typing import Optional, Tuple

class CameraStream:
    def __init__(self, color_topic: str, depth_topic: str, color_info_topic: str, depth_info_topic: str, queue_size: int = 2):
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.color_info_topic = color_info_topic
        self.depth_info_topic = depth_info_topic
        self.image_queue: Queue = Queue(maxsize=queue_size)
        self.display_thread = Thread(target=self.display_loop, daemon=True)  # Make thread daemon
        self.running = False
        self.skip_count = 0
        self.current_color_frame: Optional[np.ndarray] = None
        self.current_depth_frame: Optional[np.ndarray] = None
        self.frame_callbacks: List[Callable] = []
        self.bridge = CvBridge()
        self.color_info: Optional[CameraInfo] = None
        self.depth_info: Optional[CameraInfo] = None

    def color_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_color_frame = cv_image
            self.update_queue()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (Color): {e}")

    def depth_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.current_depth_frame = cv_image
            self.update_queue()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error (Depth): {e}")

    def color_info_callback(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def depth_info_callback(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def update_queue(self) -> None:
        if self.current_color_frame is not None and self.current_depth_frame is not None:
            if self.image_queue.full():
                self.skip_count += 1
                try:
                    self.image_queue.get_nowait()
                except:
                    pass
            try:
                self.image_queue.put_nowait((self.current_color_frame.copy(), 
                                           self.current_depth_frame.copy()))
            except:
                pass

    def process_frame(
        self,
        color_frame: np.ndarray,
        depth_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            self.current_color_frame = color_frame.copy()
            self.current_depth_frame = depth_frame.copy()
        
            for callback in self.frame_callbacks:
                callback(self.current_color_frame, self.current_depth_frame, self.color_info, self.depth_info)

            return color_frame, depth_frame
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")
            return None, None

    def display_loop(self) -> None:
        while self.running and not rospy.is_shutdown():
            if not self.image_queue.empty():
                try:
                    color_frame, depth_frame = self.image_queue.get_nowait()
                    self.process_frame(color_frame, depth_frame)
                except Queue.Empty:
                    continue

        cv2.destroyAllWindows()

    def add_frame_callback(self, callback: Callable) -> None:
        self.frame_callbacks.append(callback)

    def run(self) -> None:
        self.running = True
        self.display_thread.start()
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_info_topic, CameraInfo, self.color_info_callback)
        rospy.Subscriber(self.depth_info_topic, CameraInfo, self.depth_info_callback)


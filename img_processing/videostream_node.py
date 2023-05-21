import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from picamera2 import Picamera2

width = 384
height = width

class VideoCapture(Node):
    def __init__(self):
        super().__init__('video_capture')
        self.publisher = self.create_publisher(Image, 'video_frames', 10)
        # Timer callback is triggered 60 times per second
        timer_period = 1/60
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.picam2 = Picamera2()
        # Configure picamera2 to capture video with the format and size expected by the model.
        # This way we avoid having to reformat and resize later
        # raw is set to 1640x1232 to force high FOV sensor mode.
        # FrameDurationLimits is the min and max frame period in microseconds, 40000us limits capture
        # to 25 FPS
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'RGB888', "size": (width, height)},
                                                                     raw={"size": (1640,1232)},
                                                                     controls={"FrameDurationLimits": (40000, 40000)}))
        self.picam2.start()
        self.br = CvBridge()

    def timer_callback(self):
        frame = self.picam2.capture_array()

        # Cv bridge wraps the image array in a ROS image message
        self.publisher.publish(self.br.cv2_to_imgmsg(frame))
        self.get_logger().info('Video frame published')


def main(args=None):
    # Initialize ROS client library
    rclpy.init(args=args)

    # Instantiate the video_capture node
    video_capture = VideoCapture()
    
    # Spin node forever
    rclpy.spin(video_capture)
    video_capture.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
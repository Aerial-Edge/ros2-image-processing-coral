import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from picamera2 import Picamera2

width = 384
height = width

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, 'video_frames', 10)
        timer_period = 1/60
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}, raw={"size": (1640,1232)}, controls={"FrameDurationLimits": (40000, 40000)}))
        self.picam2.start()
        self.br = CvBridge()

    def timer_callback(self):
        frame = self.picam2.capture_array()
        #frame = cv2.resize(frame, (320, 320))
        self.publisher.publish(self.br.cv2_to_imgmsg(frame))
        self.get_logger().info('Video frame published')


def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
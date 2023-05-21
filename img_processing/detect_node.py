import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from picamera2 import Picamera2



# Absolute path to tflite model
model ='/home/gruppe6/models/edl1_1k_edgetpu.tflite'

# Load edgetpu runtime shared library
tpu_interpreter = tflite.Interpreter(model, experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1.0')])

cpu_interpreter = tflite.Interpreter(model)
# We discard detections with lower confidence score than the threshold
threshold = 0.80

class Detect(Node):
    def __init__(self):
        # Instantiate parent class object (Node)
        super().__init__('detect')

        # Create subscription to video_frames topic
        self.subscription = self.create_subscription(Image, 'video_frames', self.listener_callback, 10)

        # Create publisher to object_pos_and_distance topic
        self.publisher = self.create_publisher(Int32MultiArray, 'object_pos_and_distance', 10)

        # cv_bridge is used to extract the image array from the ROS image msg
        self.br = CvBridge()

        # Selected interpreter, change to cpu_interpreter to run
        # model on cpu only
        self.interpreter = tpu_interpreter

        # Allocate tensors, must be called to start inference
        self.interpreter.allocate_tensors()

        # Input details for the loaded model
        self.input_details = self.interpreter.get_input_details()

        # Output details for the loaded model
        self.output_details = self.interpreter.get_output_details()

        self.period_timer_start = time.time()
        self.period_timer_end = time.time()
        self.fps = 0
        # Multiplier used for distance calculation
        self.focal_length_multiplier = 847 # pi camera 2

        self.ball_real_diameter = 6.5 # cm
        # ROS2 message type, used to publish to 'object pos and distance'
        self.output_array = Int32MultiArray()

        # Get image dimensions expected by the model
        self.width = self.input_details[0]['shape'][1]
        self.height = self.input_details[0]['shape'][2]
        
    # Returns False if val is outside cutoff, used to discard
    # distance measurements at the edges of the frame
    def constrain_detection(self, val, frame_dim, cutoff):
        return val > cutoff and val < (frame_dim - cutoff)
    
    # Callback for video_frames subscription, called every 
    # time a new video frame is recieved
    def listener_callback(self, data):
        self.period_timer_end = time.time()
        self.timer_period = self.period_timer_end - self.period_timer_start
        self.fps = 1 / self.timer_period
        self.period_timer_start = time.time()
        self.get_logger().info('Current FPS: {:.2f}'.format(self.fps))
        current_frame = self.br.imgmsg_to_cv2(data)

        
        # Add batch dimension expected by the model,
        # new shape = [1, WIDTH, HEIGHT, 3]
        input_data = np.expand_dims(current_frame, axis=0)

        # Set input_data as the model input
        self.interpreter.set_tensor(self.input_details[0]['index'],
                                    input_data)
        
        # Start inference by invoking the interpreter
        self.interpreter.invoke()

        # boxes, classes, scores and number of detections are the
        # available model outputs
        # We are only using boxes and scores in this node
        
        # output_details returns the tensor index needed by get_tensor()
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]




        # Loop over all detections
        for i in range(len(scores)):
            if ((scores[i] > threshold) and (scores[i] <= 1.0)):
                # Get corner coordinates of bounding box
                x1, x2 = int(boxes[i][1] * self.width) , int(boxes[i][3] * self.width)
                y1, y2 = int(boxes[i][0] * self.height), int(boxes[i][2] * self.height)

                w, h = x2 - x1, y2 - y1
                # Get center of bounding box
                cx, cy = (int(x1 + 0.5*w),int(y1+0.5*h))

                # Because distance calculation is based on width, we need to discard
                # detections at the left and right edges as the width may be cropped
                if (self.constrain_detection(cx, self.width, 30)):
                    dist = int((self.ball_real_diameter * self.focal_length_multiplier) / w)
                    self.output_array.data = [cx, cy, dist]
                    self.publisher.publish(self.output_array)
                else:
                    self.output_array.data = [cx, cy, -1]
                    self.publisher.publish(self.output_array)
            else:
                self.output_array.data = [-1, -1, -1]
                self.publisher.publish(self.output_array)


                







def main(args=None):
    # Initialize ROS client library
    rclpy.init(args=args)

    # Instantiate the detect node
    detect = Detect()

    # Spin node forever
    rclpy.spin(detect)

    detect.destroy_node()
    rclpy.shutdown()
    

if (__name__ == "__main__"):
    main()

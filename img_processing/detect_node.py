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
tpu_interpreter = tflite.Interpreter(model, experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1.0')])
cpu_interpreter = tflite.Interpreter(model)
threshold = 0.80

class Detect(Node):
    def __init__(self):
        super().__init__('detect')
        self.subscription = self.create_subscription(Image, 'video_frames', self.listener_callback, 10)
        self.subscription
        self.publisher = self.create_publisher(Int32MultiArray, 'object_pos_and_distance', 10)
        self.br = CvBridge()

        # Selected interpreter, change to cpu_interpreter to run model on cpu only
        self.interpreter = tpu_interpreter

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.period_timer_start = time.time()
        self.period_timer_end = time.time()
        self.fps = 0
        self.focal_length = 847 # pi camera 2
        self.ball_real_diameter = 6.5 # cm
        self.output_array = Int32MultiArray()

        # Get image dimensions expected by the model
        self.width = self.input_details[0]['shape'][1]
        self.height = self.input_details[0]['shape'][2]
        
    # Returns False if val is outside cutoff, used to discard distance measurements at the edges of the frame
    def constrain_detection(self, val, frame_dim, cutoff):
        return val > cutoff and val < (frame_dim - cutoff)

    def listener_callback(self, data):
        self.period_timer_end = time.time()
        self.timer_period = self.period_timer_end - self.period_timer_start
        self.fps = 1 / self.timer_period
        self.period_timer_start = time.time()
        self.get_logger().info('Recieving video frame, current FPS: {:.2f}'.format(self.fps))
        current_frame = self.br.imgmsg_to_cv2(data)

        
        # Add batch dimension expected by the model, new shape = [1, WIDTH, HEIGHT, 3]
        input_data = np.expand_dims(current_frame, axis=0)

        # Set input_data as the model input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Start inference by invoking the interpreter
        self.interpreter.invoke()

        # boxes, classes, scores and number of detections are the
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        #classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        #num_detections = self.interpreter.get_tensor(self.output_details[2]['index'])[0]



        # for i in range(len(scores)):
        #     if ((scores[i][0] > threshold) and (scores[i][0] <= 1.0)):
        #         print(scores[i][0])

        #msg = Int32MultiArray()

        if (len(scores) != 0):
            if ((scores[0] > threshold) and (scores[0] <= 1.0)):
                x1, x2, y1, y2 = int(boxes[0][1] * self.width) , int(boxes[0][3] * self.width), int(boxes[0][0] * self.height), int(boxes[0][2] * self.height)
                w, h = x2 - x1, y2 - y1
                cx, cy = (int(x1 + 0.5*w),int(y1+0.5*h))
                #box_diagonal_length = int(np.sqrt(w**2 + h**2))
                if (self.constrain_detection(cx, self.width, 0) and self.constrain_detection(cy, self.height, 0)):
                    dist = int((self.ball_real_diameter * self.focal_length) / w) # distance in cm
                    self.output_array.data = [cx, cy, dist]
                    self.publisher.publish(self.output_array)
                else:
                    self.output_array.data = [-1, -1, -1]
                    self.publisher.publish(self.output_array)
            else:
                self.output_array.data = [-1, -1, -1]
                self.publisher.publish(self.output_array)


                







def main(args=None):
    rclpy.init(args=args)
    detect = Detect()
    rclpy.spin(detect)
    detect.destroy_node()
    rclpy.shutdown()
    

if (__name__ == "__main__"):
    main()

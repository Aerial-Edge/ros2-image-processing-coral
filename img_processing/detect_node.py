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



#model ='../../share/img_processing/models/tennis_edgetpu.tflite'
model ='/home/gruppe6/models/edl0.tflite'
#tpu_interpreter = tflite.Interpreter(model, experimental_delegates=[
#    tflite.load_delegate('libedgetpu.so.1.0')])
cpu_interpreter = tflite.Interpreter(model)
threshold = 0.4


class Detect(Node):
    def __init__(self):
        super().__init__('detect')
        self.subscription = self.create_subscription(Image, 'video_frames', self.listener_callback, 10)
        self.subscription
        self.publisher = self.create_publisher(Int32MultiArray, 'object_pos_and_distance', 10)
        self.br = CvBridge()
        self.interpreter = cpu_interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #self.cap = cv2.VideoCapture(0)
        #self.ret, self.current_frame_raw = self.cap.read()
        #self.raw_height, self.raw_width, _ = self.current_frame_raw.shape
        self.is_first_message = True
        self.frame_count = 0
        self.period_timer_start = time.time()
        self.period_timer_end = time.time()
        self.fps = 0
        

        

    def listener_callback(self, data):
        self.frame_count += 1
        self.period_timer_end = time.time()
        self.timer_period = self.period_timer_end - self.period_timer_start
        self.fps = 1 / self.timer_period
        self.period_timer_start = time.time()
        self.get_logger().info('Recieving video frame, current FPS: {:.2f}'.format(self.fps))
        current_frame = self.br.imgmsg_to_cv2(data)

        if (self.is_first_message):
            self.raw_height, self.raw_width, _ = current_frame.shape
            self.is_first_message = False
        
        
        #Reformat input data
        #current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        #current_frame_resized = cv2.resize(current_frame_rgb, (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
        input_data = np.expand_dims(current_frame, axis=0)



        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        #classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]



        for i in range(len(scores)):
            if ((scores[i] > threshold) and (scores[i] <= 1.0)):
                x1, x2, y1, y2 = int(boxes[i][1] * self.raw_width) , int(boxes[i][3] * self.raw_width), int(boxes[i][0] * self.raw_height), int(boxes[i][2] * self.raw_height)
                w, h = x2 - x1, y2 - y1
                c = (int(x1 + 0.5*w),int(y1+0.5*h))
                #cv2.line(current_frame, c, c, (0,255,0), 6)
                #cv2.rectangle(current_frame, (x1,y1), (x2,y2),(255,0,0), 3)
                #cv2.rectangle(current_frame_raw, (2,2), (100,100),(255,0,0), 3)
                #print(scores[i])
                #print("width: ", w, "height: ", h)
                #print ("x1: ",x1, "y1: ", y1)
                #print("Inference time", interpreter_elapsed_time)

    #     cv2.imshow('live', current_frame_raw)
    #     if (cv2.waitKey(1) == ord('q')):
    #         break

        if (len(scores) != 0):
            if ((scores[0] > threshold) and (scores[0] <= 1.0)):
                x1, x2, y1, y2 = int(boxes[0][1] * self.raw_width) , int(boxes[0][3] * self.raw_width), int(boxes[0][0] * self.raw_height), int(boxes[0][2] * self.raw_height)
                w, h = x2 - x1, y2 - y1
                cx, cy = (int(x1 + 0.5*w),int(y1+0.5*h))
                box_diagonal_length = int(np.sqrt(w**2 + h**2))


                msg = Int32MultiArray()
                msg.data = [cx, cy, box_diagonal_length]
                self.publisher.publish(msg)


    # cap.release()
    # cv2.destroyAllWindows()




def main(args=None):
    rclpy.init(args=args)
    detect = Detect()
    rclpy.spin(detect)
    detect.destroy_node()
    rclpy.shutdown()
    

if (__name__ == "__main__"):
    main()

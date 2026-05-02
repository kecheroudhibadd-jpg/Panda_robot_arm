import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

# Initialisation de MediaPipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

def get_finger_angle(a, b, c):
    """Calcule l'angle de flexion d'un doigt entre 3 points landmarks"""
    radians = math.atan2(c.y - b.y, c.x - b.x) - \
              math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(math.degrees(radians)) % 360
    return angle

def map_value(val, in_min, in_max, out_min, out_max):
    """Convertit une valeur d'une plage vers une autre plage"""
    val = max(in_min, min(in_max, val))
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.cap = cv2.VideoCapture(0)
        self.joint_angles = [0.0] * 7

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # joint1 : position X de la main
            self.joint_angles[0] = map_value(lm[9].x, 0.0, 1.0, -2.8, 2.8)
            # joint2 : position Y de la main
            self.joint_angles[1] = map_value(lm[9].y, 0.0, 1.0, -1.7, 1.7)
            # joint3 : angle du pouce
            a_pouce = get_finger_angle(lm[0], lm[2], lm[4])
            self.joint_angles[2] = map_value(a_pouce, 0, 180, -2.8, 2.8)
            # joint4 : angle de l'index
            a_index = get_finger_angle(lm[0], lm[6], lm[8])
            self.joint_angles[3] = map_value(a_index, 0, 180, -3.0, -0.1)
            # joint5 : angle de l'auriculaire
            a_auriculaire = get_finger_angle(lm[0], lm[18], lm[20])
            self.joint_angles[4] = map_value(a_auriculaire, 0, 180, -2.8, 2.8)
            # joint6 et joint7 : fixes
            self.joint_angles[5] = 0.0
            self.joint_angles[6] = 0.0

            cv2.putText(frame, f"J1: {self.joint_angles[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"J2: {self.joint_angles[1]:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"J3(pouce): {self.joint_angles[2]:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"J4(index): {self.joint_angles[3]:.2f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"J5(auriculaire): {self.joint_angles[4]:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Hand Control - FR3", frame)
        cv2.waitKey(1)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'fr3_joint{i+1}' for i in range(7)]
        msg.position = self.joint_angles
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = HandControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

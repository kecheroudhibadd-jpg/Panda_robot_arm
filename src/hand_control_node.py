import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import cv2
import mediapipe as mp
import numpy as np

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control_node')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

        # FR3 joint names
        self.joint_names = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
        ]

        # Joint limits (min, max) in radians for FR3
        self.joint_limits = [
            (-2.7437, 2.7437),
            (-1.7837, 1.7837),
            (-2.9007, 2.9007),
            (-3.0421, -0.1518),
            (-2.8065, 2.8065),
            (0.5445, 4.5169),
        ]

        self.current_positions = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5]
        self.get_logger().info('Hand Control Node started.')

    def get_finger_curl(self, landmarks, tip, pip):
        """Returns curl value 0.0 (open) to 1.0 (closed)"""
        tip_y = landmarks[tip].y
        pip_y = landmarks[pip].y
        curl = tip_y - pip_y
        return float(np.clip(curl * 10, 0.0, 1.0))

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark

            # Finger tip and pip indices
            fingers = [
                (8,  6),   # index  → joint1
                (12, 10),  # middle → joint2
                (16, 14),  # ring   → joint3
                (20, 18),  # pinky  → joint4
                (4,  3),   # thumb  → joint5
            ]

            curls = [self.get_finger_curl(lm, tip, pip) for tip, pip in fingers]

            # Map curl (0→1) to joint range
            positions = []
            for i, curl in enumerate(curls):
                lo, hi = self.joint_limits[i]
                pos = lo + curl * (hi - lo)
                positions.append(pos)

            # joint6 mirrors joint1
            positions.append(positions[0])
            self.current_positions = positions

            self.mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

        # Publish joint states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions
        self.publisher.publish(msg)

        cv2.imshow('Hand Control', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
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

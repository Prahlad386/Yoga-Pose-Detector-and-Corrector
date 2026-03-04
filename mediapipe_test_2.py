import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import math
import time

class PoseCorrectionEngine:
    def __init__(self):
        # Changed to VIDEO mode for tracking stability
        self.base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        
        self.pose_landmarks_list = []
        # State tracking variables
        self.state = "START" # START, HOLDING, RESTING
        self.start_time = 0
        self.hold_duration = 4.0
        self.rest_duration = 2.0
        self.round_counter = 0
        
        self.pose_engine_video()

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if not detection_result.pose_landmarks:
            return rgb_image
        
        self.pose_landmarks_list = detection_result.pose_landmarks[0]
        annotated_image = np.copy(rgb_image)
        
        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in detection_result.pose_landmarks:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)
        return annotated_image

    def cos_rule(self, a, b, c):
        # Added a clamp to prevent math domain errors if landmarks overlap
        val = (a**2 + b**2 - c**2) / (2*a*b + 1e-6)
        # return math.acos(max(-1, min(1, val)))
        return math.acos(val)

    def is_arms_down(self):
        """Checks if hands are resting near thighs."""
        # Wrist y-coordinate should be greater than Hip y-coordinate
        left_down = math.fabs(self.pose_landmarks_list[15].y - self.pose_landmarks_list[23].y) < 0.05
        right_down = math.fabs(self.pose_landmarks_list[16].y - self.pose_landmarks_list[24].y) < 0.05
        return left_down and right_down

    def correction_engine(self):
        if not self.pose_landmarks_list: return False
        
        # --- UNIT SCALING ---
        # Instead of raw pixels, we use shoulder width as a 'unit' (approx 15-18 inches)
        # This makes 0.16 roughly equivalent to 6 inches for an average adult.
        shoulder_width = math.hypot(self.pose_landmarks_list[11].x - self.pose_landmarks_list[12].x, 
                                    self.pose_landmarks_list[11].y - self.pose_landmarks_list[12].y)

        feet_check = False
        shoulder_ear_check_left = False
        shoulder_ear_check_right = False
        finger_check = False
        elbow_check_left = False
        elbow_check_right = False
        knee_check_left = False
        knee_check_right = False

        # 1. Feet Check (Using your logic: Distance < 0.16 relative to frame)
        if math.fabs(self.pose_landmarks_list[29].y - self.pose_landmarks_list[30].y) < 0.05 and \
           math.fabs((self.pose_landmarks_list[29].x - self.pose_landmarks_list[30].x) - (shoulder_width*0.4)) < 0.1:
            feet_check = True

        # 2. Arms near Ears Check
        if math.hypot(self.pose_landmarks_list[11].y - self.pose_landmarks_list[7].y, self.pose_landmarks_list[11].x - self.pose_landmarks_list[7].x) < 0.12:
            shoulder_ear_check_left = True
        if math.hypot(self.pose_landmarks_list[12].y - self.pose_landmarks_list[8].y, self.pose_landmarks_list[12].x - self.pose_landmarks_list[8].x) < 0.12:
            shoulder_ear_check_right = True

        # 3. Finger/Palm Join Check
        if math.hypot(self.pose_landmarks_list[17].y - self.pose_landmarks_list[18].y, self.pose_landmarks_list[17].x - self.pose_landmarks_list[18].x) < 0.08:
            finger_check = True

        # 4. Knee Checks (Straight legs)
        def check_straight(p1, p2, p3):
            a = math.hypot(self.pose_landmarks_list[p1].y - self.pose_landmarks_list[p2].y, self.pose_landmarks_list[p1].x - self.pose_landmarks_list[p2].x)
            b = math.hypot(self.pose_landmarks_list[p2].y - self.pose_landmarks_list[p3].y, self.pose_landmarks_list[p2].x - self.pose_landmarks_list[p3].x)
            c = math.hypot(self.pose_landmarks_list[p1].y - self.pose_landmarks_list[p3].y, self.pose_landmarks_list[p1].x - self.pose_landmarks_list[p3].x)
            return math.fabs(self.cos_rule(a, b, c) - math.pi) < 0.25 # Increased threshold for usability

        knee_check_left = check_straight(23, 25, 27)
        knee_check_right = check_straight(24, 26, 28)
        
        # 5. Elbow Checks (Straight arms)
        elbow_check_left = check_straight(11, 13, 15)
        elbow_check_right = check_straight(12, 14, 16)

        joint_list = [feet_check, shoulder_ear_check_left, shoulder_ear_check_right, finger_check, elbow_check_left, elbow_check_right, knee_check_left, knee_check_right]
        return all(joint_list)

    def pose_engine_video(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # Prepare Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int(time.time() * 1000)
            
            # Detect
            detection_result = self.detector.detect_for_video(mp_image, timestamp)
            annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
            display_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            if detection_result.pose_landmarks:
                pose_correct = self.correction_engine()
                
                # --- STATE MACHINE ---
                current_time = time.time()
                
                if self.state == "START":
                    status_text = "Assume Pose: Arms Up!"
                    if pose_correct:
                        self.state = "HOLDING"
                        self.start_time = current_time

                elif self.state == "HOLDING":
                    elapsed = current_time - self.start_time
                    status_text = f"HOLDING: {elapsed:.1f}/4.0s"
                    if not pose_correct:
                        self.state = "START" # Reset if form breaks
                    elif elapsed >= self.hold_duration:
                        self.state = "TRANSITION"

                elif self.state == "TRANSITION":
                    status_text = "Great! Now lower arms to thighs."
                    if self.is_arms_down():
                        self.state = "RESTING"
                        self.start_time = current_time
                
                elif self.state == "COMPLETE":
                    status_text = "Exercise Complete! Well done!"

                elif self.state == "RESTING":
                    elapsed = current_time - self.start_time
                    status_text = f"RESTING: {elapsed:.1f}/2.0s"
                    if elapsed >= self.rest_duration:
                        self.round_counter+=1
                        if self.round_counter == 3:
                            status_text = "ROUND 3 EXERCISE COMPLETE!"
                            self.state = "COMPLETE"
                        else:
                            status_text = f"ROUND {self.round_counter} COMPLETE!"
                            # Reset for next rep
                            #if not self.is_arms_down():
                            self.state = "START"

                cv2.putText(display_frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Tadasana Yoga Correction Engine", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = PoseCorrectionEngine()
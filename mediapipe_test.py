import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import cv2
import math

class PoseCorrectionEngine:
  def __init__(self):
    self.base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    self.options = vision.PoseLandmarkerOptions(
        base_options=self.base_options,
        output_segmentation_masks=True)
    self.pose_landmarks_list = []
    self.detector = vision.PoseLandmarker.create_from_options(self.options)
    self.pose_engine()


  def draw_landmarks_on_image(self, rgb_image, detection_result):
    print('entered function')
    pose_landmarks_list = detection_result.pose_landmarks
    self.pose_landmarks_list = pose_landmarks_list[0]
    annotated_image = np.copy(rgb_image)
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=pose_landmarks,
          connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
          landmark_drawing_spec=pose_landmark_style,
          connection_drawing_spec=pose_connection_style)

    return annotated_image
  
  def cos_rule(self, a, b, c):
    return math.acos((a**2 + b**2 - c**2) / (2*a*b))

  def correction_engine(self):
    print('entered correction engine')
    feet_check = False
    shoulder_ear_check_left = False
    shoulder_ear_check_right = False
    finger_check = False
    # wrist_check_right = False
    # wrist_check_left = False
    elbow_check_right = False
    elbow_check_left = False
    knee_check_left = False
    knee_check_right = False

    #ALL THE CHECKS!
    if math.fabs(self.pose_landmarks_list[29].y - self.pose_landmarks_list[30].y) < 0.01 and math.fabs(self.pose_landmarks_list[29].x - self.pose_landmarks_list[30].x) < 0.16:
      feet_check = True
      print('feet check passed')
    if math.hypot(self.pose_landmarks_list[11].y - self.pose_landmarks_list[7].y, self.pose_landmarks_list[11].x - self.pose_landmarks_list[7].x) < 0.02:
      shoulder_ear_check_left = True
      print('shoulder ear check left passed')
    if math.hypot(self.pose_landmarks_list[12].y - self.pose_landmarks_list[8].y, self.pose_landmarks_list[12].x - self.pose_landmarks_list[8].x) < 0.02:
      shoulder_ear_check_right = True
      print('shoulder ear check right passed')
    if math.hypot(self.pose_landmarks_list[17].y - self.pose_landmarks_list[18].y, self.pose_landmarks_list[17].x - self.pose_landmarks_list[18].x) < 0.01:
      finger_check = True
      print('finger check passed')
    a = math.hypot(self.pose_landmarks_list[23].y - self.pose_landmarks_list[25].y, self.pose_landmarks_list[23].x - self.pose_landmarks_list[25].x)
    b = math.hypot(self.pose_landmarks_list[25].y - self.pose_landmarks_list[27].y, self.pose_landmarks_list[25].x - self.pose_landmarks_list[27].x)
    c = math.hypot(self.pose_landmarks_list[23].y - self.pose_landmarks_list[27].y, self.pose_landmarks_list[23].x - self.pose_landmarks_list[27].x)
    if math.fabs(self.cos_rule(a, b, c) - math.pi) < 0.1:
      print('knee check left passed')
      knee_check_left = True
    else:
      print('knee check left failed')
    d = math.hypot(self.pose_landmarks_list[24].y - self.pose_landmarks_list[26].y, self.pose_landmarks_list[24].x - self.pose_landmarks_list[26].x)
    e = math.hypot(self.pose_landmarks_list[26].y - self.pose_landmarks_list[28].y, self.pose_landmarks_list[26].x - self.pose_landmarks_list[28].x)
    f = math.hypot(self.pose_landmarks_list[24].y - self.pose_landmarks_list[28].y, self.pose_landmarks_list[24].x - self.pose_landmarks_list[28].x)
    if math.fabs(self.cos_rule(d, e, f) - math.pi) < 0.1:
      print('knee check right passed')
      knee_check_right = True
    else:
      print('knee check right failed')
    
    a1 = math.hypot(self.pose_landmarks_list[11].y - self.pose_landmarks_list[13].y, self.pose_landmarks_list[11].x - self.pose_landmarks_list[13].x)
    b1 = math.hypot(self.pose_landmarks_list[13].y - self.pose_landmarks_list[15].y, self.pose_landmarks_list[13].x - self.pose_landmarks_list[15].x)
    c1 = math.hypot(self.pose_landmarks_list[11].y - self.pose_landmarks_list[15].y, self.pose_landmarks_list[11].x - self.pose_landmarks_list[15].x)
    if math.fabs(self.cos_rule(a1, b1, c1) - math.pi) < 0.1:
      print('elbow check left passed')
      elbow_check_left = True
    else:
      print('elbow check left failed')
    d1 = math.hypot(self.pose_landmarks_list[12].y - self.pose_landmarks_list[14].y, self.pose_landmarks_list[12].x - self.pose_landmarks_list[14].x)
    e1 = math.hypot(self.pose_landmarks_list[14].y - self.pose_landmarks_list[16].y, self.pose_landmarks_list[14].x - self.pose_landmarks_list[16].x)
    f1 = math.hypot(self.pose_landmarks_list[12].y - self.pose_landmarks_list[16].y, self.pose_landmarks_list[12].x - self.pose_landmarks_list[16].x)
    if math.fabs(self.cos_rule(d1, e1, f1) - math.pi) < 0.1:
      print('elbow check right passed')
      elbow_check_right = True
    else:
      print('elbow check right failed')
    joint_list = [feet_check, shoulder_ear_check_left, shoulder_ear_check_right, finger_check, elbow_check_left, elbow_check_right, knee_check_left, knee_check_right]
    for i in joint_list:
      if i == False:
        print('overall check failed')
        return False
    print('overall check passed')
    return True
    
  def pose_engine(self):
    print('entered pose engine')
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("image2.jpeg")

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = self.detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
    t = self.correction_engine()
    if t:
      print('the yoga pose is correct')
    else:
      print('the yoga pose is incorrect')
    # img_final = cv2.imread(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print('gonna show image')
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":  
  engine = PoseCorrectionEngine()
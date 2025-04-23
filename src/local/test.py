from deepface import DeepFace
from pathlib import Path

dfs = DeepFace.find(
    img_path='../db',
    db_path=''
)














# import cv2
# import onnxruntime as ort
# import numpy as np
#
# recognizer = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', '')
#
#
# def preprocess(img):
#     resized = cv2.resize(img, (112, 112))
#     normed = resized.astype(np.float32) / 255.0
#     normed = (normed - 0.5) / 0.5  # normalize to [-1, 1]
#     return np.transpose(normed, (2, 0, 1))[None, :, :, :]  # NCHW
#
#
# # Load model
# session = ort.InferenceSession("mobilefacenet.onnx")
# input_name = session.get_inputs()[0].name
#
# # Camera
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     face_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_tensor = preprocess(face_img)
#     embedding = session.run(None, {input_name: input_tensor})[0]
#
#     print("Face embedding:", embedding[:, :5])  # print first 5 dims
#     cv2.imshow("Face", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

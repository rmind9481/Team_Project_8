#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 필요한 라이브러리 import
import cgi, os, sys, codecs
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image, ImageFile
import io
import re
import cgitb
cgitb.enable()  # 에러 발생 시 브라우저에서 확인 가능

# CGI 출력을 위한 한글 깨짐 방지
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 손상된 이미지도 허용

# HTML 응답 헤더 출력 (꼭 필요함)
print("Content-Type: text/html; charset=utf-8\n")

# ✅ 기본 경로 설정 (절대경로 사용으로 안정성 향상)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(base_dir, 'model.pkl')         # 학습된 모델 경로
upload_dir = os.path.join(base_dir, 'uploads')           # 업로드 이미지 저장 경로
result_dir = os.path.join(base_dir, 'results')           # 결과 이미지 저장 경로

# 폴더가 없으면 자동으로 생성
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


# 요청 방식 확인: GET이면 업로드 폼 출력
if os.environ.get("REQUEST_METHOD", "GET") == "GET":
    print("""
    <!DOCTYPE html>
    <html lang="ko">
           
    <head><meta charset="UTF-8"><title>이미지 업로드</title></head>
    
    <body>
        <h2>🖼️ 카리나를 맞춰보세요~ </h2>
        <img src="../uploads/carina_(100).jpg" width="300" alt="Karina"><br><br>

        <form enctype="multipart/form-data" method="post">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="예측하기">
        </form>
    </body>
    </html>
    """)
    sys.exit()



# ✅ 모델 불러오기
try:
    model = joblib.load(model_path)
except:
    print("<h2>❌ model.pkl 파일을 불러올 수 없습니다.</h2>")
    sys.exit()

# 얼굴 검출기 로드 (Haar Cascade 방식)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ HOG 특징 추출 함수 정의
def extract_hog_features(img, size=(128, 128)):
    # 이미지를 128x128로 리사이즈 후 HOG 추출
    img = cv2.resize(img, size)
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

# ✅ 업로드된 이미지 수신 (form에서 받기)
form = cgi.FieldStorage()

# 업로드 필드 "image"가 존재하는지 확인
if "image" not in form:
    print("<h2>❗ 업로드된 이미지가 없습니다.</h2>")
    sys.exit()

# 파일 이름 추출
fileitem = form["image"]
filename = os.path.basename(fileitem.filename)
if not filename:
    print("<h2>❗ 파일 이름이 없습니다.</h2>")
    sys.exit()

# 업로드된 이미지 저장
filepath = os.path.join(upload_dir, filename)
file_bytes = fileitem.file.read()  # 바이트로 읽기
with open(filepath, 'wb') as f:
    f.write(file_bytes)

# ✅ PIL로 이미지 열기 및 OpenCV 변환
try:
    # PIL로 이미지 열기 (검증 포함)
    img_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_np = np.array(img_pil)  # PIL → NumPy 배열
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # RGB → BGR(OpenCV용)
except Exception as e:
    print(f"<h2>❌ 이미지 처리 실패: {e}</h2>")
    sys.exit()

# ✅ 얼굴 검출 (흑백 이미지에서 진행)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 얼굴이 없으면 메시지 출력
if len(faces) == 0:
    print("<h2>😢 얼굴이 감지되지 않았습니다.</h2>")
else:
    # 얼굴이 있을 경우 반복 처리
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]  # 얼굴만 잘라냄
        features = extract_hog_features(face_img)  # HOG 특징 추출
        pred = model.predict([features])[0]        # 예측 (0 또는 1)
        label = "Carina" if pred == 1 else "Others"
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)

        # 사각형과 예측 결과 라벨을 이미지에 표시
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ✅ 결과 이미지 파일명 안전하게 만들기
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    result_filename = "result_" + safe_filename
    result_path = os.path.join(result_dir, result_filename)

    # ✅ OpenCV가 저장 가능한 타입으로 변환 (혹시 모를 타입 오류 방지)
    img_to_save = img.copy()
    if img_to_save.dtype != np.uint8:
        img_to_save = (255 * img_to_save).astype(np.uint8) if img_to_save.max() <= 1 else img_to_save.astype(np.uint8)

    # ✅ 이미지 저장 시도
    success = cv2.imwrite(result_path, img_to_save)
    if not success:
        print("<h2>❌ 결과 이미지 저장 실패 (cv2.imwrite 실패)</h2>")
        sys.exit()

    # ✅ 결과 이미지 웹에 출력
    print(f"""
    <h2>🎯 얼굴 인식 결과</h2>
  
    <img src="../results/{result_filename}" width="400"><br><br>
    <a href="../cgi-bin/face_detect.py">← 다시 업로드하기</a>
    """)

from flask import Flask, request, send_file
from flask_restx import Api, Resource
import cv2
import numpy as np
import easyocr
import re
from io import BytesIO
from flask_cors import CORS

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)
api = Api(app, version="1.0", title="OCR Masking API",
          description="API for masking personal information in images")

# EasyOCR 모델 및 Haar Cascade 초기화
reader = easyocr.Reader(['ko', 'en'])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Swagger에 사용할 모델 정의
upload_parser = api.parser()
upload_parser.add_argument('image', location='files', type='file', required=True, help='Image file to process')

def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def mask_personal_info(img):
    img = resize_image(img)
    results = reader.readtext(img)

    personal_info_patterns = [
        r"\d{6}[-\s]?\d{1,7}",
        r"\b\d{2,3}-\d{3,4}-\d{4}\b",
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    ]

    address_keywords = [
        "서울", "특별시", "도", "광역시", "구", "군", "시", "읍", "면", "동", "리", "번지",
        "로", "길", "아파트", "건물", "타워", "빌딩", "블럭", "호", "층", "주택", "리버", "파크", "산",
        "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남",
        "전북", "전남", "경북", "경남", "제주", "김포", "양산", "창원", "포항", "안산", "용인",
        "수원", "성남", "고양", "청주", "천안", "전주", "여수", "순천", "평택", "의정부", "춘천", "남양주"
    ]

    full_text = " ".join([text for (_, text, _) in results])

    # 개인정보 패턴 블러 처리
    for pattern in personal_info_patterns:
        matches = re.finditer(pattern, full_text)
        for match in matches:
            matched_text = match.group()
            for (bbox, text, _) in results:
                if matched_text in text:
                    (top_left, bottom_right) = (tuple(map(int, bbox[0])), tuple(map(int, bbox[2])))
                    img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.blur(
                        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (51, 51)
                    )

    # 주소 키워드 블러 처리
    for keyword in address_keywords:
        if keyword in full_text:
            for (bbox, text, _) in results:
                if keyword in text:
                    (top_left, bottom_right) = (tuple(map(int, bbox[0])), tuple(map(int, bbox[2])))
                    img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.blur(
                        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (51, 51)
                    )

    # 얼굴 블러 처리
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (51, 51))

    return img

@api.route('/process_image')
@api.expect(upload_parser)
class ProcessImage(Resource):
    def post(self):
        try:
            args = request.files['image']
            image_bytes = np.frombuffer(args.read(), np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if img is None:
                app.logger.error("Invalid image input.")
                return {"error": "Invalid image file."}, 400

            # 마스킹 처리
            masked_img = mask_personal_info(img)

            # 마스킹된 이미지를 JPEG로 변환
            _, buffer = cv2.imencode('.jpg', masked_img)

            # 메모리에서 파일로 반환
            return send_file(BytesIO(buffer), mimetype='image/jpeg')
        except Exception as e:
            app.logger.exception("Error processing image.")
            return {"error": "An error occurred while processing the image."}, 500

# 엔드포인트를 API에 등록
api.add_namespace(api, path="/")

if __name__ == '__main__':
    app.run(debug=True)

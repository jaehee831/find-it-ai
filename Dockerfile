# Python 3.9 경량 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 파일 복사
COPY requirements.txt .
COPY OCR_name.py .

# 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Flask 앱 실행 명령어
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "OCR_name:app"]

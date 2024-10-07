from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
import mysql.connector
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
import os
import shutil
import cv2
import numpy as np
import base64

# AI 모델 관련 임포트
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn

app = Flask(__name__)
app.secret_key = 'suhan1029'  # 보안을 위해 실제 서비스에서는 환경 변수로 관리하세요.

# 세션을 서버 측 파일 시스템에 저장하도록 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False  # 영구 세션 비활성화

# Flask-Session 초기화
Session(app)

# 서버 시작 시 세션 파일 삭제
if os.path.exists(app.config['SESSION_FILE_DIR']):
    shutil.rmtree(app.config['SESSION_FILE_DIR'])
os.makedirs(app.config['SESSION_FILE_DIR'])

# MySQL 데이터베이스 연결 설정
db = mysql.connector.connect(
    host="localhost",
    user="root",           # MySQL 사용자 이름
    password="1029",       # MySQL 비밀번호
    database="user_info"   # 사용하시는 데이터베이스 이름으로 변경하세요.
)

# 로그인 필요 데코레이터
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# AI 모델 클래스 이름
class_names = ['fall', 'spring', 'summer', 'winter']

# 모델 로드 함수
def load_model(model_path):
    model = timm.create_model('efficientnet_b4', pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25020723272413975),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

# 모델 로드 (앱 시작 시 한 번만 로드)
model = load_model('personal_color_efficientnet_b4_76.pth')  # 모델 파일 경로 확인

# 메인 페이지
@app.route('/')
def home():
    user_id = session.get('user_id')
    return render_template('first.html', user_id=user_id)

# 회원가입 페이지
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        email = request.form['email']
        id_checked = request.form.get('id_checked')

        if not id_checked:
            flash('아이디 중복 확인을 해주세요.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        cursor = db.cursor()
        sql = "INSERT INTO users (user_id, password, email) VALUES (%s, %s, %s)"
        cursor.execute(sql, (user_id, hashed_password, email))
        db.commit()
        flash('회원가입이 완료되었습니다.')
        return redirect(url_for('login'))
    return render_template('register.html')

# 아이디 중복 확인 AJAX 요청 처리
@app.route('/check_id', methods=['POST'])
def check_id():
    user_id = request.form['user_id']
    cursor = db.cursor()
    sql = "SELECT * FROM users WHERE user_id = %s"
    cursor.execute(sql, (user_id,))
    result = cursor.fetchone()
    if result:
        return '중복'
    else:
        return '사용 가능'

# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']

        cursor = db.cursor()
        sql = "SELECT password FROM users WHERE user_id = %s"
        cursor.execute(sql, (user_id,))
        result = cursor.fetchone()
        if result and check_password_hash(result[0], password):
            session['user_id'] = user_id
            flash(f'{user_id}님, 환영합니다!')
            return redirect(url_for('home'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
    return render_template('login.html')

# 로그아웃
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('로그아웃되었습니다.')
    return redirect(url_for('home'))

# 퍼스널 컬러 진단 페이지 (웹캠 이미지 업로드)
@app.route('/personal_color', methods=['GET', 'POST'])
@login_required
def personal_color():
    if request.method == 'POST':
        # 웹캠을 통해 캡처한 이미지 수신
        image_data = request.form['image_data']
        image_data = image_data.split(',')[1]  # base64 헤더 제거
        image_data = base64.b64decode(image_data)
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # OpenCV를 사용한 이미지 처리
        cv2.imshow('Captured Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # PIL로 이미지 변환 후 예측
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess_image(image)
        prediction = predict(image_tensor, model)

        return render_template('result.html', prediction=prediction, user_id=session.get('user_id'))
    return render_template('index.html', user_id=session.get('user_id'))

if __name__ == '__main__':
    app.run(debug=True)

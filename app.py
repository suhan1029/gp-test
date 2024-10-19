from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
import mysql.connector
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
from dotenv import load_dotenv
import os
import shutil
from openai import OpenAI
import logging
from werkzeug.utils import secure_filename
from io import BytesIO
import uuid
import markdown
import bleach

# AI 모델 관련 임포트
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", 'your_secret_key')  # 보안을 위해 환경 변수로 관리하세요.

# 세션을 서버 측 파일 시스템에 저장하도록 설정
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False  # 영구 세션 비활성화

# Flask-Session 초기화
Session(app)

# 업로드된 이미지를 저장할 디렉토리 설정
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# MySQL 데이터베이스 연결 설정
db = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),           # MySQL 사용자 이름
    password=os.getenv("DB_PASSWORD", ""),       # MySQL 비밀번호
    database=os.getenv("DB_NAME", "user_info")   # 사용하시는 데이터베이스 이름으로 변경하세요.
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

@app.route('/about_personal_color')
def about_personal_color():
    return render_template('about_personal_color.html', user_id=session.get('user_id'))

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
        cursor.close()
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
    cursor.close()
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
        cursor.close()
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

# 퍼스널 컬러 진단 페이지 (이미지 업로드)
@app.route('/personal_color', methods=['GET', 'POST'])
@login_required
def personal_color():
    if request.method == 'POST':
        # 파일 업로드 처리
        if 'file' not in request.files:
            flash('파일이 없습니다.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('파일이 선택되지 않았습니다.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = Image.open(file).convert('RGB')
            image_tensor = preprocess_image(image)
            prediction = predict(image_tensor, model)

            # 이미지 파일 저장
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            # OpenAI API를 사용하여 설명 생성
            prompt = f"사용자의 얼굴 이미지에서 '{prediction}' 퍼스널 컬러가 나왔습니다. '{prediction}' 퍼스널 컬러의 특징과 사용자의 이미지에서 그 퍼스널 컬러가 나온 이유를 간단하게 설명해줘."

            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "당신은 퍼스널 컬러 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ]
                )
                explanation_markdown = completion.choices[0].message.content.strip()
                explanation_html = markdown.markdown(explanation_markdown)
                allowed_tags = ['p', 'strong', 'em', 'ul', 'ol', 'li', 'br', 'code', 'pre', 'blockquote']
                explanation_html = bleach.clean(explanation_html, tags=allowed_tags, strip=True)

            except Exception as e:
                error_message = f'이미지 분석 중 오류가 발생했습니다: {str(e)}'
                flash(error_message)
                logging.error(error_message)
                explanation_markdown = "죄송합니다. 이미지를 분석할 수 없습니다."
                explanation_html = "<p>죄송합니다. 이미지를 분석할 수 없습니다.</p>"

            # 예측 결과 및 설명을 데이터베이스에 저장, 데이터베이스에 저장할 때는 Markdown 원본을 저장
            cursor = db.cursor()
            sql = "INSERT INTO predictions (user_id, prediction, prediction_time, img, explains) VALUES (%s, %s, NOW(), %s, %s)"
            cursor.execute(sql, (session['user_id'], prediction, filename, explanation_markdown))
            db.commit()
            cursor.close()

            # 템플릿 렌더링 시 HTML 변환된 설명 전달
            return render_template('result.html', prediction=prediction, user_id=session.get('user_id'), explanation=explanation_html)
        else:
            flash('허용되지 않는 파일 형식입니다.')
            return redirect(request.url)
    return render_template('index.html', user_id=session.get('user_id'))

# 허용된 파일 확장자 확인 함수
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 히스토리 페이지
@app.route('/history')
@login_required
def history():
    cursor = db.cursor(dictionary=True)
    sql = "SELECT predict_id, prediction, prediction_time, img, explains FROM predictions WHERE user_id = %s ORDER BY prediction_time DESC"
    cursor.execute(sql, (session['user_id'],))
    history_records = cursor.fetchall()
    cursor.close()
    
    # Convert explanations from markdown to HTML
    allowed_tags = ['p', 'strong', 'em', 'ul', 'ol', 'li', 'br', 'code', 'pre', 'blockquote']
    for record in history_records:
        explanation_markdown = record['explains']
        explanation_html = markdown.markdown(explanation_markdown)
        explanation_html = bleach.clean(explanation_html, tags=allowed_tags, strip=True)
        record['explains_html'] = explanation_html

    return render_template('history.html', user_id=session.get('user_id'), history_records=history_records)

# 챗봇 페이지
@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    cursor = db.cursor(dictionary=True)
    # 사용자별 대화 목록 가져오기
    sql = "SELECT * FROM conversations WHERE user_id = %s ORDER BY created_at DESC"
    cursor.execute(sql, (session['user_id'],))
    conversations = cursor.fetchall()
    cursor.close()
    return render_template('chat.html', user_id=session.get('user_id'), conversations=conversations)

@app.route('/render_markdown', methods=['POST'])
def render_markdown():
    data = request.get_json()
    markdown_text = data.get('markdown', '')
    html = markdown.markdown(markdown_text)
    # 허용된 태그만 남기고 필터링
    allowed_tags = ['p', 'strong', 'em', 'ul', 'ol', 'li', 'br', 'code', 'pre', 'blockquote']
    html = bleach.clean(html, tags=allowed_tags, strip=True)
    return html

# 새로운 대화 시작 라우트
@app.route('/start_conversation', methods=['POST'])
@login_required
def start_conversation():
    conversation_name = request.form.get('conversation_name', 'New Conversation')
    cursor = db.cursor()
    sql = "INSERT INTO conversations (user_id, conversation_name) VALUES (%s, %s)"
    cursor.execute(sql, (session['user_id'], conversation_name))
    conversation_id = cursor.lastrowid
    db.commit()
    cursor.close()
    return redirect(url_for('chat_conversation', conversation_id=conversation_id))

# 특정 대화 표시 및 메시지 처리 라우트
@app.route('/chat/<int:conversation_id>', methods=['GET', 'POST'])
@login_required
def chat_conversation(conversation_id):
    cursor = db.cursor(dictionary=True)
    sql = "SELECT * FROM conversations WHERE conversation_id = %s AND user_id = %s"
    cursor.execute(sql, (conversation_id, session['user_id']))
    conversation = cursor.fetchone()

    if not conversation:
        flash('대화를 찾을 수 없습니다.')
        return redirect(url_for('chat'))
    
    # 대화 및 메시지 가져오기
    messages = get_conversation_messages(conversation_id)

    # 사용자별 대화 목록 가져오기
    sql = "SELECT * FROM conversations WHERE user_id = %s ORDER BY created_at DESC"
    cursor.execute(sql, (session['user_id'],))
    conversations = cursor.fetchall()
    cursor.close()

    if request.method == 'POST':
        user_message = request.form['message']
        # 사용자 메시지를 저장
        cursor = db.cursor()
        sql = "INSERT INTO messages (conversation_id, sender, message) VALUES (%s, %s, %s)"
        cursor.execute(sql, (conversation_id, 'user', user_message))
        db.commit()

        # OpenAI API 호출
        messages = get_conversation_messages(conversation_id)
        assistant_response_html, assistant_response_markdown = get_assistant_response(messages)

        # 어시스턴트 응답을 저장 (HTML과 Markdown 원본 모두 저장)
        sql = "INSERT INTO messages (conversation_id, sender, message, raw_markdown) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (conversation_id, 'assistant', assistant_response_html, assistant_response_markdown))
        db.commit()
        cursor.close()

        # 메시지 목록 다시 가져오기
        messages = get_conversation_messages(conversation_id)

        return render_template('conversation.html', user_id=session.get('user_id'), conversation=conversation, messages=messages, conversations=conversations, assistant_response_markdown=assistant_response_markdown)
    else:
        return render_template('conversation.html', user_id=session.get('user_id'), conversation=conversation, messages=messages, conversations=conversations)

def get_conversation_messages(conversation_id):
    cursor = db.cursor(dictionary=True)
    sql = "SELECT sender, message, raw_markdown FROM messages WHERE conversation_id = %s ORDER BY created_at ASC"
    cursor.execute(sql, (conversation_id,))
    messages = cursor.fetchall()
    cursor.close()
    return messages

def get_assistant_response(messages):
    # 메시지 포맷 구성
    formatted_messages = []
    for msg in messages:
        role = msg['sender']
        content = msg['message']
        formatted_messages.append({"role": role, "content": content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 또는 사용 가능한 모델 지정
            messages=formatted_messages
        )
        assistant_response_markdown = response.choices[0].message.content.strip()

        # Markdown을 HTML로 변환
        assistant_response_html = markdown.markdown(assistant_response_markdown)

        # Sanitize HTML
        allowed_tags = ['p', 'strong', 'em', 'ul', 'ol', 'li', 'br', 'code', 'pre', 'blockquote']
        assistant_response_html = bleach.clean(assistant_response_html, tags=allowed_tags, strip=True)
        return assistant_response_html, assistant_response_markdown
    
    except Exception as e:
        logging.error(f'OpenAI API 호출 중 오류 발생: {str(e)}')
        return "죄송합니다. 응답을 생성할 수 없습니다."

if __name__ == '__main__':
    app.run(debug=True)
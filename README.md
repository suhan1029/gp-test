# gp-test

졸업작품(graduation project)를 간단히 만들고 테스트한 자료입니다.

로컬에서 실행하기 위해 파일을 다운받을 때 zip 파일로 한번에 다운받은 다음 AI 모델은 별도로 RAW 파일을 다운받고 zip 파일로 받은 모델 파일은 삭제해야합니다.

<br>

## OpenCV 관련

opencv 기능을 넣은 것과 원본을 구분하기 위해 opencv 파일에는 opencv라고 표시해뒀습니다.

실행할때는 파일 이름을 원본과 똑같이 바꾸고 실행해야 합니다.

<br>

## MySQL DB 생성

커넥션 설정은 app.py 코드를 보고 해주시고 테이블 설정은 다음 과정을 따릅니다.

CREATE DATABASE user_info;

USE user_info;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL
);

# gp-test

졸업작품(graduation project) 제작소입니다.

로컬에서 실행하기 위해 파일을 다운받을 때 zip 파일로 한번에 다운받은 다음 AI 모델은 별도로 RAW 파일을 다운받고 zip 파일로 받은 모델 파일은 삭제해야합니다.

<br>

## 생성형 이미지 관련

- OpenAI API를 활용하여 이미지 생성 기능을 구현하였습니다.

- API 키를 생성하기 위해서는 ChatGPT가 아닌 OpenAI 사이트로 들어가야 합니다.

- API 키는 보안을 위해 환경변수로 감춰두었습니다.
  - API 키는 작업 폴더내에 '.env'라는 파일을 만들고 다음과 같은 내용을 입력하면 됩니다.
    ```
    OPENAI_API_KEY=your_api_key
    ```


- 참고한 사이트 - https://platform.openai.com/docs/guides/images/usage
- ~~현재 API가 입력한 프롬프트를 가볍게 무시하고 계십니다.~~
- ~~망할 GPT o1-preview가 API 사용 방법을 잘 모릅니다.~~

### api 키 생성하는 방법

- 사이트 접속 - https://platform.openai.com/docs/overview
- 화면 좌측 API keys 클릭
- API 생성
    

<br>

## OpenCV 관련

- OpenCV 관련 기능들을 기존의 app.py에 합쳐놓았습니다.

<br>

## MySQL DB 생성

- 커넥션 설정은 app.py 코드를 보고 해주시고 테이블 설정은 다음 과정을 따릅니다.
  ```
  CREATE DATABASE user_info;
  ```
  
  ```
  USE user_info;
  ```
  
  ```
  CREATE TABLE users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(255) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      email VARCHAR(255) NOT NULL
  );
  ```
  
  ```
  CREATE TABLE predictions (
      predict_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(255),
      prediction VARCHAR(50),
      prediction_time DATETIME,
      FOREIGN KEY (user_id) REFERENCES users(user_id)
  );
  ```















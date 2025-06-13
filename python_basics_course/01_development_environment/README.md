# 모듈 1: Python 개발 환경 설정

## 학습 목표
- Python 개발 환경을 로컬 시스템에 설정할 수 있다
- 가상 환경을 생성하고 관리할 수 있다
- 필요한 패키지를 설치하고 관리할 수 있다
- VS Code를 효율적으로 사용할 수 있다

## 준비사항
- 컴퓨터 (Windows/Mac/Linux)
- 인터넷 연결
- 관리자 권한 (설치 과정에 필요할 수 있음)

## 1. Python 설치하기

### Windows
1. [Python 공식 웹사이트](https://www.python.org/downloads/)에서 최신 버전 다운로드
2. 설치 파일 실행 (중요: "Add Python to PATH" 옵션 체크)
3. 설치 완료 후 확인:
   ```bash
   python --version
   ```

### Mac
1. Homebrew를 사용한 설치:
   ```bash
   brew install python
   ```
2. 설치 확인:
   ```bash
   python3 --version
   ```

### Linux (Ubuntu/Debian)
1. 패키지 매니저를 통한 설치:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```
2. 설치 확인:
   ```bash
   python3 --version
   ```

## 2. 가상 환경 설정

### 가상 환경이란?
- 프로젝트별로 독립된 Python 환경을 제공
- 패키지 버전 충돌 방지
- 프로젝트 의존성 관리 용이

### 가상 환경 생성 및 관리
1. 프로젝트 디렉토리 생성:
   ```bash
   mkdir my_project
   cd my_project
   ```

2. 가상 환경 생성:
   ```bash
   python -m venv venv
   ```

3. 가상 환경 활성화:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. 가상 환경 비활성화:
   ```bash
   deactivate
   ```

## 3. 패키지 설치 및 관리

### 기본 패키지 설치
```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### requirements.txt 사용
1. 현재 설치된 패키지 목록 저장:
   ```bash
   pip freeze > requirements.txt
   ```

2. requirements.txt로부터 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 4. VS Code 설정

### 필수 확장 프로그램
1. Python (Microsoft)
2. Jupyter
3. Python Indent
4. autoDocstring

### VS Code 설정
1. Python 인터프리터 선택:
   - `Ctrl+Shift+P` (Windows/Linux) 또는 `Cmd+Shift+P` (Mac)
   - "Python: Select Interpreter" 선택
   - 생성한 가상 환경 선택

2. 편의 기능 설정:
   - 자동 저장
   - 코드 포맷팅 (Black 포매터 추천)
   - 린터 설정 (pylint 또는 flake8)

## 실습 과제

### 1. 기본 환경 설정
1. Python을 설치하고 버전을 확인하세요
2. 새 프로젝트 디렉토리를 만들고 가상 환경을 설정하세요
3. 필수 패키지들을 설치하세요

### 2. VS Code 설정
1. 추천된 확장 프로그램들을 설치하세요
2. Python 인터프리터를 가상 환경으로 설정하세요
3. 기본 settings.json 파일을 구성하세요

### 3. 테스트 코드 실행
다음 코드를 실행하여 환경이 제대로 설정되었는지 확인하세요:

```python
# test_environment.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_environment():
    print(f"Python 버전: {sys.version}")
    print(f"NumPy 버전: {np.__version__}")
    print(f"Pandas 버전: {pd.__version__}")
    
    # 간단한 데이터 시각화 테스트
    data = np.random.normal(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    sns.histplot(data)
    plt.title("정규 분포 테스트")
    plt.show()

if __name__ == "__main__":
    test_environment()
```

## 문제 해결 가이드
- Path 설정 문제: 시스템 환경 변수 확인
- 가상 환경 활성화 실패: 실행 권한 확인
- 패키지 설치 오류: pip 업그레이드 시도
- VS Code 인터프리터 인식 문제: 경로 직접 지정

## 다음 단계
환경 설정이 완료되면, 다음 모듈에서는 Python 기초 문법을 학습할 예정입니다.

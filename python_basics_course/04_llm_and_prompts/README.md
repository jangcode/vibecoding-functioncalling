# 모듈 4: LLM & OpenAI API와 Context 관리

## 학습 목표
- OpenAI API의 기본 구조와 사용 방법을 이해한다
- 효과적인 프롬프트 설계 원칙을 실습한다
- Context 관리와 토큰 비용 최적화 방법을 익힌다

## 1. LLM 기초 이해

### 1.1 LLM이란?
- 대규모 언어 모델의 정의와 특징
- 주요 LLM 모델들 (GPT, BERT, LLaMA 등)
- LLM의 활용 분야와 한계점

### 1.2 LLM의 작동 원리
- 트랜스포머 아키텍처 기초
- 토큰화와 임베딩
- 컨텍스트 윈도우
- 추론 과정의 이해

## 2. 효과적인 프롬프트 작성

### 2.1 프롬프트 엔지니어링 기본 원칙
1. 명확성 (Clarity)
   - 구체적인 지시사항 제공
   - 모호성 제거
   - 단계별 작업 분리

2. 구조화 (Structure)
   - 일관된 형식 사용
   - 논리적 순서 배열
   - 예시 포함

3. 컨텍스트 제공 (Context)
   - 배경 정보 포함
   - 원하는 출력 형식 지정
   - 제약 조건 명시

### 2.2 프롬프트 패턴

#### 역할 기반 프롬프트
```text
역할: [특정 전문가/역할]
작업: [수행해야 할 작업]
컨텍스트: [관련 배경 정보]
제약조건: [고려해야 할 제약사항]
출력형식: [원하는 응답 형식]
```

#### 단계별 프롬프트
```text
다음 단계에 따라 [작업]을 수행하시오:
1. [첫 번째 단계]
2. [두 번째 단계]
3. [세 번째 단계]
...
출력형식: [원하는 형식]
```

#### 예시 기반 프롬프트
```text
다음 예시와 같은 형식으로 [작업]을 수행하시오:

입력 예시:
[입력 데이터 예시]

출력 예시:
[원하는 출력 형식 예시]

실제 입력:
[실제 입력 데이터]
```

### 2.3 프롬프트 최적화 기법

1. 점진적 개선
   - 기본 프롬프트 작성
   - 결과 분석
   - 프롬프트 수정 및 개선
   - 반복 테스트

2. 체인 프롬프트
   - 복잡한 작업을 단계별로 분리
   - 이전 단계의 출력을 다음 단계의 입력으로 사용
   - 결과의 품질 향상

3. Few-shot 학습
   - 다양한 예시 제공
   - 패턴 학습 유도
   - 일관된 출력 형식 유지

## 3. LLM API 활용

### 3.1 OpenAI API 사용
```python
import openai

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message["content"]
```

### 3.2 주요 매개변수 이해
- Temperature
- Max tokens
- Top P
- Frequency penalty
- Presence penalty

### 3.3 에러 처리와 재시도
```python
import backoff
import openai

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_completion_with_backoff(prompt, model="gpt-3.5-turbo"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message["content"]
    except openai.error.APIError as e:
        print(f"API 오류: {e}")
        return None
```

## 실습 과제

### 1. 기본 프롬프트 작성
다음 작업을 수행하는 프롬프트를 작성하세요:
- 텍스트 요약
- 코드 리뷰
- 데이터 분석 리포트 생성
- 마케팅 문구 작성

### 2. 대화형 AI 어시스턴트 구현
```python
class AIAssistant:
    def __init__(self):
        self.conversation_history = []
        
    def add_to_history(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_input):
        # 사용자 입력 추가
        self.add_to_history("user", user_input)
        
        try:
            # API 호출
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                temperature=0.7
            )
            
            # 응답 저장
            assistant_response = response.choices[0].message["content"]
            self.add_to_history("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"오류 발생: {e}")
            return None
```

### 3. 프롬프트 최적화 프로젝트
특정 작업에 대한 프롬프트를 작성하고 다음 과정을 통해 최적화하세요:
1. 기본 프롬프트 작성
2. 결과 평가
3. 프롬프트 수정
4. A/B 테스트
5. 최종 프롬프트 선정

## 모범 사례와 주의사항

### 모범 사례
1. 명확하고 구체적인 지시사항 제공
2. 예시를 통한 기대 출력 명시
3. 단계별 작업 분리
4. 컨텍스트 충분히 제공
5. 제약조건 명확히 설정

### 주의사항
1. 개인정보 포함 주의
2. API 사용량 모니터링
3. 결과 검증 필수
4. 에러 처리 구현
5. 비용 관리

## 추가 학습 자료
- [OpenAI Documentation](https://platform.openai.com/docs/guides/completion)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Papers on LLM](https://github.com/thunlp/PLMpapers)

## 다음 단계
LLM과 프롬프트 엔지니어링의 기초를 마스터했다면, 
실제 프로젝트에서 이를 활용하여 다양한 응용 프로그램을 개발해보세요.

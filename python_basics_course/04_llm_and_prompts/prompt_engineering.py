import openai
import os
import json
import backoff
from typing import List, Dict, Optional

class PromptTemplate:
    def __init__(self, template: str):
        """프롬프트 템플릿을 초기화합니다.

        Args:
            template (str): 변수를 포함한 프롬프트 템플릿 문자열
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """템플릿의 변수를 주어진 값으로 대체합니다.

        Args:
            **kwargs: 템플릿의 변수에 대응하는 값들

        Returns:
            str: 완성된 프롬프트 문자열
        """
        return self.template.format(**kwargs)

class AIAssistant:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """AI 어시스턴트를 초기화합니다.

        Args:
            api_key (str, optional): OpenAI API 키
            model (str, optional): 사용할 모델 이름
        """
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("API 키가 필요합니다.")

        self.model = model
        self.conversation_history = []
        self.templates = self._load_default_templates()

    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """기본 프롬프트 템플릿을 로드합니다."""
        return {
            "summarize": PromptTemplate(
                "다음 텍스트를 {max_words}단어 이내로 요약해주세요:\n\n{text}"
            ),
            "code_review": PromptTemplate(
                "다음 코드를 리뷰해주세요. 특히 다음 측면들을 중점적으로 봐주세요:\n"
                "- 코드 품질\n"
                "- 가독성\n"
                "- 성능\n"
                "- 보안\n\n"
                "코드:\n```{language}\n{code}\n```"
            ),
            "data_analysis": PromptTemplate(
                "다음 데이터를 분석하고 주요 인사이트를 도출해주세요:\n\n{data}"
            )
        }

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """OpenAI API를 호출하여 응답을 받아옵니다.

        Args:
            prompt (str): 프롬프트 문자열
            temperature (float, optional): 응답의 무작위성 정도

        Returns:
            Optional[str]: AI의 응답 또는 오류 시 None
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
            return None

    def add_to_history(self, role: str, content: str):
        """대화 기록에 새로운 메시지를 추가합니다.

        Args:
            role (str): 메시지 작성자의 역할 ("user" 또는 "assistant")
            content (str): 메시지 내용
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    def get_chat_response(self, user_input: str) -> Optional[str]:
        """대화 맥락을 유지하며 응답을 생성합니다.

        Args:
            user_input (str): 사용자 입력

        Returns:
            Optional[str]: AI의 응답 또는 오류 시 None
        """
        self.add_to_history("user", user_input)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message["content"]
            self.add_to_history("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            print(f"챗봇 응답 생성 중 오류 발생: {e}")
            return None

    def summarize_text(self, text: str, max_words: int = 100) -> Optional[str]:
        """텍스트를 요약합니다.

        Args:
            text (str): 요약할 텍스트
            max_words (int, optional): 최대 단어 수

        Returns:
            Optional[str]: 요약문 또는 오류 시 None
        """
        prompt = self.templates["summarize"].format(
            text=text,
            max_words=max_words
        )
        return self.get_completion(prompt)

    def review_code(self, code: str, language: str) -> Optional[str]:
        """코드를 리뷰합니다.

        Args:
            code (str): 리뷰할 코드
            language (str): 프로그래밍 언어

        Returns:
            Optional[str]: 코드 리뷰 또는 오류 시 None
        """
        prompt = self.templates["code_review"].format(
            code=code,
            language=language
        )
        return self.get_completion(prompt)

    def analyze_data(self, data: str) -> Optional[str]:
        """데이터를 분석합니다.

        Args:
            data (str): 분석할 데이터

        Returns:
            Optional[str]: 분석 결과 또는 오류 시 None
        """
        prompt = self.templates["data_analysis"].format(data=data)
        return self.get_completion(prompt)

def main():
    """메인 실행 함수"""
    # API 키 설정 (실제 사용시에는 환경 변수나 설정 파일에서 로드)
    api_key = "your-api-key"
    
    try:
        # AI 어시스턴트 초기화
        assistant = AIAssistant(api_key)
        
        # 텍스트 요약 예제
        text = """
        인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력, 자연언어의 이해능력 등을 
        컴퓨터 프로그램으로 실현한 기술입니다. 최근에는 딥러닝의 발전으로 인해 
        이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 혁신적인 발전을 
        이루고 있습니다.
        """
        summary = assistant.summarize_text(text, max_words=50)
        print("요약문:", summary)
        
        # 코드 리뷰 예제
        code = """
        def fibonacci(n):
            if n <= 0:
                return []
            elif n == 1:
                return [0]
            result = [0, 1]
            for i in range(2, n):
                result.append(result[i-1] + result[i-2])
            return result
        """
        review = assistant.review_code(code, "python")
        print("\n코드 리뷰:", review)
        
        # 대화형 응답 예제
        chat_response = assistant.get_chat_response(
            "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?"
        )
        print("\n챗봇 응답:", chat_response)
        
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

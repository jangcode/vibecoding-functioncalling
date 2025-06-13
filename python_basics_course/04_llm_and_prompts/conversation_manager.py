import os
import openai
from dotenv import load_dotenv
from typing import List, Dict
import tiktoken

class ConversationManager:
    def __init__(self, max_turns: int = 7):
        """대화 관리자를 초기화합니다.

        Args:
            max_turns (int, optional): 유지할 최대 대화 턴 수. Defaults to 7.
        """
        # 환경 변수에서 API 키 로드
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.history = []
        self.max_turns = max_turns
        self.total_tokens = 0
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def add_message(self, role: str, content: str):
        """대화 이력에 새 메시지를 추가합니다.

        Args:
            role (str): 메시지 작성자의 역할 ('user' 또는 'assistant')
            content (str): 메시지 내용
        """
        # 토큰 수 계산
        tokens = len(self.encoder.encode(content))
        self.total_tokens += tokens
        
        # 메시지 추가
        self.history.append({
            "role": role,
            "content": content,
            "tokens": tokens
        })
        
        # 최대 턴 수 유지
        if len(self.history) > self.max_turns * 2:  # 사용자와 어시스턴트 메시지 각각 계산
            # 가장 오래된 대화 제거 및 토큰 수 업데이트
            removed_messages = self.history[:-self.max_turns * 2]
            for msg in removed_messages:
                self.total_tokens -= msg["tokens"]
            self.history = self.history[-self.max_turns * 2:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """현재 대화 이력을 API 호출에 적합한 형태로 반환합니다.

        Returns:
            List[Dict[str, str]]: OpenAI API에 전달할 메시지 목록
        """
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in self.history]
    
    def get_token_usage(self) -> Dict[str, int]:
        """현재까지의 토큰 사용량 통계를 반환합니다.

        Returns:
            Dict[str, int]: 토큰 사용량 통계
        """
        return {
            "total_tokens": self.total_tokens,
            "messages_count": len(self.history),
            "average_tokens_per_message": self.total_tokens / len(self.history) if self.history else 0
        }
    
    def clear_history(self):
        """대화 이력을 초기화합니다."""
        self.history = []
        self.total_tokens = 0

def demo_conversation():
    """대화 관리 데모를 실행합니다."""
    # 대화 관리자 초기화
    manager = ConversationManager(max_turns=7)
    
    # 테스트용 대화 시나리오
    conversation = [
        "안녕하세요!",
        "파이썬 프로그래밍을 배우고 싶어요.",
        "객체지향 프로그래밍이 뭔가요?",
        "예시 코드를 보여줄 수 있나요?",
        "상속은 어떻게 구현하나요?",
        "다중 상속의 장단점은?",
        "디자인 패턴과의 관계는?"
    ]
    
    try:
        for user_input in conversation:
            print("\n사용자:", user_input)
            
            # 사용자 입력 추가
            manager.add_message("user", user_input)
            
            # API 호출
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=manager.get_messages()
            )
            
            # 응답 처리
            assistant_response = response.choices[0].message['content']
            manager.add_message("assistant", assistant_response)
            
            print("어시스턴트:", assistant_response)
            
            # 토큰 사용량 출력
            usage = manager.get_token_usage()
            print("\n=== 토큰 사용량 통계 ===")
            print(f"총 토큰 수: {usage['total_tokens']}")
            print(f"메시지 수: {usage['messages_count']}")
            print(f"메시지당 평균 토큰 수: {usage['average_tokens_per_message']:.2f}")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    demo_conversation()

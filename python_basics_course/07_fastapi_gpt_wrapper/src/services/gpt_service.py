from fastapi import HTTPException
import openai
from ..config import get_settings
from ..models.chat import ChatRequest, ChatResponse

settings = get_settings()
openai.api_key = settings.openai_api_key

class GPTService:
    """GPT 서비스 클래스"""
    
    @staticmethod
    async def generate_chat_response(request: ChatRequest) -> ChatResponse:
        """채팅 응답을 생성합니다.

        Args:
            request (ChatRequest): 채팅 요청 객체

        Returns:
            ChatResponse: 생성된 응답과 사용량 정보

        Raises:
            HTTPException: OpenAI API 호출 중 오류 발생 시
        """
        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.model_name,
                messages=[msg.dict() for msg in request.messages],
                temperature=request.temperature or settings.temperature,
                max_tokens=request.max_tokens or settings.max_tokens
            )
            
            return ChatResponse(
                response=response.choices[0].message.content,
                usage=response.usage
            )
            
        except openai.error.RateLimitError:
            raise HTTPException(
                status_code=429,
                detail="API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            )
        except openai.error.InvalidRequestError as e:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 요청입니다: {str(e)}"
            )
        except openai.error.AuthenticationError:
            raise HTTPException(
                status_code=401,
                detail="API 키가 유효하지 않습니다."
            )
        except openai.error.OpenAIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API 오류: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"서버 오류: {str(e)}"
            )

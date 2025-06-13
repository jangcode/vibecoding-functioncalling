# 모듈 8: 인증·구독 시스템 연동 & Next.js LLM UI

## 학습 목표
- JWT 기반 인증 시스템을 구현할 수 있다
- 결제 시스템(Stripe/PayPal)을 연동할 수 있다
- Next.js로 LLM API 결과를 표시하는 UI를 구현할 수 있다

## 프로젝트 구조
```
.
├── README.md
├── backend/                     # FastAPI 백엔드
│   ├── requirements.txt        # 백엔드 패키지 목록
│   └── src/
│       ├── main.py            # FastAPI 앱
│       ├── config.py          # 설정
│       ├── auth/              # 인증 관련
│       ├── payments/          # 결제 관련
│       ├── models/            # 데이터 모델
│       └── utils/             # 유틸리티
│
└── frontend/                   # Next.js 프론트엔드
    ├── package.json
    ├── components/            # React 컴포넌트
    ├── pages/                 # Next.js 페이지
    ├── styles/                # CSS 스타일
    └── utils/                 # 유틸리티
```

## 1. 백엔드 설정

### 1.1 필요 패키지 설치
```bash
# backend/requirements.txt
fastapi>=0.68.0
uvicorn>=0.15.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.5
sqlalchemy>=1.4.23
pydantic>=2.0.0
stripe>=2.60.0
paypalrestsdk>=1.13.1
python-dotenv>=0.19.0
```

### 1.2 JWT 인증 설정
```python
# backend/src/auth/jwt.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# JWT 설정
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### 1.3 결제 시스템 연동
```python
# backend/src/payments/stripe_handler.py
import stripe
from fastapi import APIRouter, HTTPException

router = APIRouter()
stripe.api_key = "your-stripe-secret-key"

@router.post("/create-checkout-session")
async def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': 'price_H5ggYwtDq4fbrJ',
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:3000/success',
            cancel_url='http://localhost:3000/cancel',
        )
        return {"sessionId": session.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/webhook")
async def webhook_received(stripe_signature: str):
    try:
        event = stripe.Webhook.construct_event(
            payload=request.body,
            sig_header=stripe_signature,
            secret="your-webhook-secret"
        )
        
        if event.type == "checkout.session.completed":
            # 결제 완료 처리
            pass
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## 2. 프론트엔드 설정

### 2.1 Next.js 프로젝트 생성
```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint
cd frontend
npm install @stripe/stripe-js axios jsonwebtoken
```

### 2.2 인증 컴포넌트
```typescript
// frontend/components/auth/LoginForm.tsx
import { useState } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

export default function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/auth/login', {
        email,
        password,
      });
      
      localStorage.setItem('token', response.data.token);
      router.push('/dashboard');
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
          Email
        </label>
        <input
          type="email"
          id="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        />
      </div>
      <div>
        <label htmlFor="password" className="block text-sm font-medium text-gray-700">
          Password
        </label>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        />
      </div>
      <button
        type="submit"
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
      >
        Login
      </button>
    </form>
  );
}
```

### 2.3 결제 컴포넌트
```typescript
// frontend/components/payments/SubscriptionButton.tsx
import { loadStripe } from '@stripe/stripe-js';
import axios from 'axios';

const stripePromise = loadStripe('your-publishable-key');

export default function SubscriptionButton() {
  const handleSubscribe = async () => {
    try {
      const stripe = await stripePromise;
      const response = await axios.post('/api/create-checkout-session');
      
      if (stripe) {
        const result = await stripe.redirectToCheckout({
          sessionId: response.data.sessionId,
        });
        
        if (result.error) {
          console.error(result.error);
        }
      }
    } catch (error) {
      console.error('Subscription failed:', error);
    }
  };

  return (
    <button
      onClick={handleSubscribe}
      className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
    >
      Subscribe Now
    </button>
  );
}
```

### 2.4 LLM Chat 컴포넌트
```typescript
// frontend/components/chat/ChatInterface.tsx
import { useState } from 'react';
import axios from 'axios';

export default function ChatInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    try {
      setLoading(true);
      const response = await axios.post('/api/chat', {
        messages: [...messages, { role: 'user', content: input }],
      });
      
      setMessages(prev => [
        ...prev,
        { role: 'user', content: input },
        { role: 'assistant', content: response.data.response },
      ]);
      setInput('');
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg ${
              msg.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>
      
      <div className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 rounded-lg border p-2"
            placeholder="Type your message..."
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

## 3. 실습 과제

### 1. 인증 시스템 구현
1. JWT 인증 API 구현
2. 로그인/회원가입 폼 구현
3. 인증 상태 관리
4. 보호된 라우트 구현

### 2. 결제 시스템 연동
1. Stripe 결제 페이지 연동
2. 결제 웹훅 처리
3. 구독 상태 관리
4. 결제 내역 조회 구현

### 3. Chat UI 구현
1. 채팅 인터페이스 디자인
2. API 연동
3. 메시지 상태 관리
4. 로딩/에러 상태 처리

## 구동 방법

### 백엔드
```bash
# backend 디렉토리에서
pip install -r requirements.txt
uvicorn src.main:app --reload
```

### 프론트엔드
```bash
# frontend 디렉토리에서
npm install
npm run dev
```

## API 문서
- 백엔드: http://localhost:8000/docs
- 프론트엔드: http://localhost:3000

## 참고 자료
- [Next.js 문서](https://nextjs.org/docs)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Stripe 문서](https://stripe.com/docs)
- [JWT 인증 가이드](https://jwt.io/)

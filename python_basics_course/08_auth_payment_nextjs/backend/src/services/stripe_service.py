import stripe
from decouple import config
from fastapi import HTTPException, status

stripe.api_key = config("STRIPE_SECRET_KEY")

class StripeService:
    @staticmethod
    async def create_checkout_session(price_id: str, user_id: int):
        try:
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url='http://localhost:3000/payment/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url='http://localhost:3000/payment/cancel',
                client_reference_id=str(user_id),
            )
            return checkout_session
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    @staticmethod
    async def handle_webhook(payload, sig_header):
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, config("STRIPE_WEBHOOK_SECRET")
            )
            
            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                # 구독 상태 업데이트 로직
                await update_subscription_status(
                    user_id=session.client_reference_id,
                    status="active"
                )
                
            return event
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

async def update_subscription_status(user_id: str, status: str):
    """사용자의 구독 상태를 업데이트하는 함수"""
    # 데이터베이스 업데이트 로직 구현
    pass

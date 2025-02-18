import json
import openai
from django.http import JsonResponse
from django.views import View
from django.views.generic import TemplateView
from chatbot.services import get_recommendation
import logging

# 로그 설정
logger = logging.getLogger(__name__)

# OpenAI API 키 설정
openai.api_key = 'your_openai_api_key'  # OpenAI API 키를 입력하세요

class ChatbotHomeView(TemplateView):
    """
    챗봇 초기 화면을 렌더링하는 뷰
    """
    template_name = "chatbot/home.html"  # 템플릿 파일 경로

class ChatAPIView(View):
    """
    GPT와 전략 추천 기능을 통합한 뷰
    """

    def post(self, request, *args, **kwargs):
        try:
            # 1. 요청 데이터 파싱
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            if not user_message:
                return JsonResponse({"response": "메시지를 입력해주세요."}, status=400)

            # 2. 메시지 처리: 유저가 "전략"을 입력하면, GPT로부터 전략을 받아서 번호 추천
            if user_message.lower() in ["start", "시작"]:
                return JsonResponse({
                    'response': "안녕하세요! 로또 추천 챗봇입니다. 전략을 선택해주세요:\n1. 많이 나온 번호 기반 추천\n2. 적게 나온 번호 기반 추천"
                }, status=200)

            # 3. GPT로부터 전략을 결정하도록 처리
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=f"유저 메시지: {user_message} \nGPT, 유저가 선택한 전략을 알려줘.",
                max_tokens=100,
                temperature=0.7,
            )

            gpt_response = response.choices[0].text.strip()

            # 4. GPT가 전략을 선택한 후, 해당 전략에 맞는 번호 추천
            if "전략 1" in gpt_response:
                recommended_numbers = get_recommendation(strategy=1)
                return JsonResponse({
                    'response': f"전략 1로 추천된 번호: {', '.join(map(str, recommended_numbers))}"
                }, status=200)

            elif "전략 2" in gpt_response:
                recommended_numbers = get_recommendation(strategy=2)
                return JsonResponse({
                    'response': f"전략 2로 추천된 번호: {', '.join(map(str, recommended_numbers))}"
                }, status=200)

            else:
                return JsonResponse({
                    'response': "전략을 정확히 선택해주세요. 전략 1 또는 전략 2를 입력해주세요."
                }, status=400)

        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API 호출 중 에러 발생: {str(e)}")
            return JsonResponse({'error': f'OpenAI API 호출 중 문제가 발생했습니다: {str(e)}'}, status=500)

        except Exception as e:
            logger.error(f"서버 에러 발생: {str(e)}")
            return JsonResponse({'error': '서버에서 문제가 발생했습니다. 관리자에게 문의하세요.'}, status=500)

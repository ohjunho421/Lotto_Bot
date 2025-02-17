# views.py

import json
import logging
import random
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from django.http import JsonResponse
from django.views import View
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.conf import settings
from chatbot.services import get_recommendation, check_data_status

logger = logging.getLogger(__name__)

class ChatbotHomeView(TemplateView):
    """View for rendering the chatbot home page"""
    template_name = 'chatbot/home.html'

class CSRFTokenView(View):
    """View for getting CSRF token"""
    def get(self, request, *args, **kwargs):
        csrf_token = get_token(request)
        return JsonResponse({'csrfToken': csrf_token})

class DataStatusView(View):
    """View for checking data status"""
    def get(self, request, *args, **kwargs):
        success, message = check_data_status()
        return JsonResponse({
            'success': success,
            'message': message
        })

@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):
    """Main API view for handling chat interactions"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.lucky_messages = [
            "행운이 함께하길 바랍니다! ✨",
            "이번에는 좋은 결과가 있기를 기원합니다! 🍀",
            "당신의 꿈이 이루어지길 바라며 이 번호들을 선택했습니다! 🌟",
            "이 번호들과 함께 큰 행운이 찾아오길 바랍니다! 🎯",
            "당첨의 기쁨을 누리실 수 있기를 진심으로 응원합니다! ⭐",
            "이번 주는 특별한 행운이 함께하길 바랍니다! 🌈",
            "당신의 성공을 기원하며 이 번호들을 추천해드립니다! 💫",
            "모든 세트에 행운이 가득하길 기원합니다! 🌠",
            "이 번호들이 당신에게 좋은 기운을 가져다주길 바랍니다! 🎊",
            "당첨의 행운이 함께하시길 진심으로 바랍니다! 💫"
        ]

    def _process_strategy_counts(self, user_message):
        """Parse strategy counts from user message"""
        strategy_counts = {'1': 0, '2': 0}
        
        try:
            message = user_message.lower()
            
            # 숫자 추출 (마지막 숫자만)
            count = 0
            numbers = ''.join(c for c in message if c.isdigit())
            if len(numbers) >= 1:
                # 전략 번호를 제외한 숫자 추출
                if message.startswith(('전략1', '전략2', '1번', '2번')):
                    count = int(numbers[1:]) if len(numbers) > 1 else 0
                else:
                    count = int(numbers)

            # 전략 확인
            if any(pattern in message for pattern in ["전략1", "전략 1", "1번 전략", "1번전략"]):
                strategy_counts['1'] = count
            elif any(pattern in message for pattern in ["전략2", "전략 2", "2번 전략", "2번전략"]):
                strategy_counts['2'] = count
            
            logger.info(f"Processed strategy counts: {strategy_counts}")
            return strategy_counts
                
        except Exception as e:
            logger.error(f"Error processing strategy counts: {e}")
            return strategy_counts

    def _format_recommendations(self, recommendations, strategy_num, num_sets):
        """Format lottery number recommendations with better readability"""
        number_sets = []
        for i, (strategy, numbers) in enumerate(recommendations, 1):
            number_sets.append(f"□ {i}세트: {', '.join(map(str, numbers))}")
        
        lucky_message = random.choice(self.lucky_messages)
        
        formatted_message = f"""[{strategy_num}번 전략의 번호를 추천해드리겠습니다]

====================================

{chr(10).join(number_sets)}

====================================

▶ {lucky_message}"""
        return formatted_message

    def _get_gpt_response(self, user_message):
        """Get response from GPT API"""
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            system_prompt = """
안녕하세요! 로또 번호 추천 챗봇입니다.

두 가지 전략으로 번호를 추천해드릴 수 있습니다:

1. 자주 당첨된 번호 기반 추천
2. 잠재력 있는 번호 기반 추천

원하시는 전략을 선택해주세요! 
최대 5세트까지 추천 가능합니다.

(예: "전략1로 3세트 추천해주세요" 또는 "전략1 2세트, 전략2 3세트 추천해주세요")
"""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except (APIError, APITimeoutError, RateLimitError, Exception) as gpt_error:
            logger.error(f"GPT Error: {str(gpt_error)}")
            error_message = "죄송합니다. 서버 연결에 문제가 발생했습니다."
            if isinstance(gpt_error, RateLimitError):
                error_message = "죄송합니다. 잠시 후 다시 시도해주세요."
            raise Exception(error_message)

    def post(self, request, *args, **kwargs):
        """Handle POST requests"""
        try:
            # Parse request data
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            logger.info(f"Received message: {user_message}")

            if not user_message:
                logger.warning("Empty message received")
                return JsonResponse({'response': '메시지를 입력해주세요.'}, status=400)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})

            try:
                # Get GPT response
                assistant_message = self._get_gpt_response(user_message)
                logger.info(f"GPT Response: {assistant_message}")
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
            except Exception as gpt_error:
                logger.error(f"GPT Error: {str(gpt_error)}", exc_info=True)
                return JsonResponse({'response': str(gpt_error)}, status=500)

            # Process strategy counts if needed
            if "전략" in user_message.lower():
                try:
                    strategy_counts = self._process_strategy_counts(user_message)
                    logger.info(f"Processed strategy counts: {strategy_counts}")
                    total_sets = sum(strategy_counts.values())
                    logger.info(f"Total sets: {total_sets}")

                    if total_sets == 0:
                        logger.warning("No sets requested")
                        return JsonResponse({
                            'response': '세트 수를 정확히 입력해주세요. (예: "전략1로 3세트 추천해주세요")'
                        }, status=400)
                    
                    # 각 전략별로 세트 수 체크
                    for strategy, count in strategy_counts.items():
                        if count > 5:
                            logger.warning(f"Strategy {strategy} requested {count} sets (exceeds limit)")
                            return JsonResponse({
                                'response': '죄송합니다. 한 번에 최대 5세트까지 추천 가능합니다. 다시 요청해주시겠어요?'
                            }, status=200)
                    
                    # 전체 세트 수가 6 이상인 경우만 체크
                    if total_sets > 5:
                        logger.warning(f"Total sets {total_sets} exceeds limit")
                        return JsonResponse({
                            'response': '죄송합니다. 최대 5세트까지만 추천 가능합니다.\n전략1과 전략2를 조합해서 5세트를 추천해드릴까요?\n(예: "전략1 3세트, 전략2 2세트")'
                        }, status=200)

                    recommendations, error = get_recommendation(strategy_counts)
                    logger.info(f"Recommendations received: {recommendations}, Error: {error}")
                    
                    if error:
                        logger.error(f"Recommendation error: {error}")
                        return JsonResponse({'response': error}, status=400)

                    if not recommendations:
                        logger.error("Empty recommendations received")
                        return JsonResponse({'response': '번호 추천 중 오류가 발생했습니다.'}, status=400)

                    # 어떤 전략을 사용했는지 확인
                    strategy_num = '1' if strategy_counts['1'] > 0 else '2'
                    num_sets = strategy_counts[strategy_num]
                    logger.info(f"Using strategy {strategy_num} for {num_sets} sets")
                    
                    assistant_message = self._format_recommendations(recommendations, strategy_num, num_sets)
                    logger.info("Successfully formatted recommendations")
                
                except ValueError as ve:
                    logger.error(f"Value Error in processing strategy: {str(ve)}", exc_info=True)
                    return JsonResponse({
                        'response': '세트 수를 정확히 입력해주세요. (예: "전략1로 3세트 추천해주세요")'
                    }, status=400)
                except Exception as e:
                    logger.error(f"Error in processing strategy: {str(e)}", exc_info=True)
                    return JsonResponse({
                        'response': '번호 추천 처리 중 오류가 발생했습니다.'
                    }, status=400)

            return JsonResponse({'response': assistant_message}, status=200)

        except json.JSONDecodeError as je:
            logger.error(f"JSON Decode Error: {str(je)}", exc_info=True)
            return JsonResponse({
                'response': '잘못된 요청 형식입니다.'
            }, status=400)
        except Exception as e:
            logger.error(f"Error in ChatAPIView: {str(e)}", exc_info=True)
            return JsonResponse({
                'response': '서버 에러가 발생했습니다. 잠시 후 다시 시도해주세요.',
                'error': str(e)
            }, status=500)
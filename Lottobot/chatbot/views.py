from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import Chatbot
from .serializers import ChatbotSerializer
from .rag_utils import retrieve_relevant_lotto_info

import openai, os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 프롬프트 명령
prompt = "너는 로또 전문가야, 추천해달라고하면 최근당첨데이터를 기반으로 예상당첨번호와 함께 추천이유도 설명해줘, 필요하다면 통계적 방법을 이용해서 해도돼"


class ChatbotAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        chat_history = Chatbot.objects.filter(author=request.user).order_by("-created_at")
        serializer = ChatbotSerializer(chat_history, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = ChatbotSerializer(data=request.data)
        if serializer.is_valid():
            user_input = serializer.validated_data["user_input"]
            chat_history = Chatbot.objects.filter(author=request.user).order_by("-created_at")
            
            message = [{"role": "system", "content": prompt}]
            
            for chat in chat_history:
                message.append({"role": "user", "content": chat.user_input})
                message.append({"role": "assistant", "content": chat.ai_response})
            
            message.append({"role": "user", "content": user_input})
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=message
            )
            ai_response = response['choices'][0]['message']['content']
            
            chatbot_instance = serializer.save(author=request.user, ai_response=ai_response)
            
            return Response(ChatbotSerializer(chatbot_instance).data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

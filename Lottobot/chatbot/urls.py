from django.urls import path
from .views import ChatAPIView, get_csrf_token, ChatbotHomeView

app_name = "chatbot"

urlpatterns = [
    path("chat/", ChatAPIView.as_view(), name="chat_api"),
    path("csrf/", get_csrf_token, name="get_csrf_token"),   # CSRF 토큰 반환
    path("", ChatbotHomeView.as_view(), name="home"),  # 홈 화면
]

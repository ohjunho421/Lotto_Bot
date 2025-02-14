from django.contrib import admin
from django.urls import path, include
from chatbot.views import ChatbotHomeView  # ChatbotHomeView 가져오기

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", ChatbotHomeView.as_view(), name="home"),  # 기본 URL에 ChatbotHomeView 연결
    path("api/chatbot/", include("chatbot.urls")),     # ChatGPT API 연결
]

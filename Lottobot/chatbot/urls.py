from django.urls import path
from . import views


app_name = "chatbot"
urlpatterns = [
    path("", views.ChatbotAPIView.as_view(), name="chatbot"),
]

from django.urls import path
from .views import ChatbotView

urlpatterns = [
    # path("top-lotto-numbers/", TopLottoNumbersView.as_view(), name="top_lotto_numbers"),
    path("chatbot1/", ChatbotView.as_view(), name="Chatbot"),
]

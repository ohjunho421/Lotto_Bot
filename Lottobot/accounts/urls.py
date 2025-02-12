from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import SignupAPIView, LogoutAPIView, DeleteAPIView


app_name = "accounts"
urlpatterns = [
    path("signup/", SignupAPIView.as_view(), name="signup"),
    path("signin/", TokenObtainPairView.as_view(), name="TokenObtainPairView"),
    path("logout/", LogoutAPIView.as_view(), name="logout"),
    path("delete/", DeleteAPIView.as_view(), name="delete"),
    # path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]

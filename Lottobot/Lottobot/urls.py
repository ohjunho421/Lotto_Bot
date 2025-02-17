from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('chatbot.urls')),  # chatbot 앱의 모든 URL을 포함
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
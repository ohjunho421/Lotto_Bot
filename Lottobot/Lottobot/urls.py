from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('main/', login_required(views.main_view), name='main'),
    path('chatbot/', include('chatbot.urls')),
    path('mypage/', login_required(views.mypage_view), name='mypage'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
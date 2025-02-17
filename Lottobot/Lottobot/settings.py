from pathlib import Path
from datetime import timedelta
import os
from dotenv import load_dotenv
from celery.schedules import crontab

# BASE_DIR 정의: settings.py 파일의 최상단에 위치해야 합니다.
BASE_DIR = Path(__file__).resolve().parent.parent

# .env 파일 로드
load_dotenv(os.path.join(BASE_DIR, '.env'))

# LOTTO_DATA_DIR와 LOTTO_DATA_FILE 정의
LOTTO_DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(LOTTO_DATA_DIR, exist_ok=True)
LOTTO_DATA_FILE = os.path.join(LOTTO_DATA_DIR, 'lotto_history.csv')

# API 키 가져오기
OPENAI_API_KEY = os.getenv('OPEN_API_KEY')

# 기본 Django 설정
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
DEBUG = True
ALLOWED_HOSTS = []

# CSRF 설정
CSRF_TRUSTED_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:3000"]

# 커스텀 유저 모델
AUTH_USER_MODEL = 'accounts.User'

# 애플리케이션 정의
INSTALLED_APPS = [
   # Django Apps
   'django.contrib.admin',
   'django.contrib.auth',
   'django.contrib.contenttypes',
   'django.contrib.sessions',
   'django.contrib.messages',
   'django.contrib.staticfiles',

   # Third Party Apps
   'rest_framework',
   'rest_framework_simplejwt',
   'rest_framework_simplejwt.token_blacklist',
   'corsheaders',

   # Local Apps
   'accounts',
   'chatbot.apps.ChatbotConfig',
   #'django_celery_beat',
]

# Celery 설정
# CELERY_BROKER_URL = 'redis://localhost:6379/0'
# CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
# CELERY_ACCEPT_CONTENT = ['json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
# CELERY_TIMEZONE = 'Asia/Seoul'

# # Celery Beat 스케줄 설정
# CELERY_BEAT_SCHEDULE = {
#    'crawl-lotto-numbers': {
#        'task': 'chatbot.tasks.update_lotto_data',
#        'schedule': crontab(day_of_week='saturday', hour='22', minute='0'),
#    },
#    'train-ml-model': {
#        'task': 'chatbot.tasks.train_ml_model',
#        'schedule': crontab(day_of_week='saturday', hour='22', minute='30'),
#    },
# }

# 미들웨어 설정
MIDDLEWARE = [
   'django.middleware.security.SecurityMiddleware',
   'corsheaders.middleware.CorsMiddleware',
   'django.contrib.sessions.middleware.SessionMiddleware',
   'django.middleware.common.CommonMiddleware',
   'django.middleware.csrf.CsrfViewMiddleware',
   'django.contrib.auth.middleware.AuthenticationMiddleware',
   'django.contrib.messages.middleware.MessageMiddleware',
   'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# URL 설정
ROOT_URLCONF = 'Lottobot.urls'

# 템플릿 설정
TEMPLATES = [
   {
       'BACKEND': 'django.template.backends.django.DjangoTemplates',
       'DIRS': [BASE_DIR / 'templates'],
       'APP_DIRS': True,
       'OPTIONS': {
           'context_processors': [
               'django.template.context_processors.debug',
               'django.template.context_processors.request',
               'django.contrib.auth.context_processors.auth',
               'django.contrib.messages.context_processors.messages',
           ],
       },
   },
]

# WSGI 설정
WSGI_APPLICATION = 'Lottobot.wsgi.application'

# 데이터베이스 설정
DATABASES = {
   'default': {
       'ENGINE': 'django.db.backends.sqlite3',
       'NAME': BASE_DIR / 'db.sqlite3',
   }
}

# 패스워드 검증 설정
AUTH_PASSWORD_VALIDATORS = [
   {
       'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
   },
   {
       'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
   },
]

# REST Framework 설정
REST_FRAMEWORK = {
   'DEFAULT_AUTHENTICATION_CLASSES': (
       'rest_framework_simplejwt.authentication.JWTAuthentication',
   ),
   'DEFAULT_PERMISSION_CLASSES': (
       'rest_framework.permissions.IsAuthenticated',
   ),
}

# JWT 설정
SIMPLE_JWT = {
   'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
   'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
   'ROTATE_REFRESH_TOKENS': True,
   'BLACKLIST_AFTER_ROTATION': True,
   'UPDATE_LAST_LOGIN': False,
   'ALGORITHM': 'HS256',
   'SIGNING_KEY': SECRET_KEY,
   'AUTH_HEADER_TYPES': ('Bearer',),
   'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
}

# 로깅 설정
LOGGING = {
   'version': 1,
   'disable_existing_loggers': False,
   'handlers': {
       'console': {
           'class': 'logging.StreamHandler',
       },
   },
   'root': {
       'handlers': ['console'],
       'level': 'INFO',
   },
   'loggers': {
       'django': {
           'handlers': ['console'],
           'level': 'INFO',
           'propagate': False,
       },
       'chatbot': {
           'handlers': ['console'],
           'level': 'INFO',
           'propagate': False,
       },
   },
}

# CORS 설정
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# 국제화 설정
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# 정적 파일 설정
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
   os.path.join(BASE_DIR, 'static'),
]

# 기본 키 필드 타입 설정
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab
from django.conf import settings

# Django 설정 모듈 지정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')

# Celery 앱 생성
app = Celery('Lottobot')

# namespace='CELERY'는 모든 셀러리 관련 구성 키를 의미합니다.
# 반드시 CELERY라는 접두사로 시작해야 합니다.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Celery Beat 설정
app.conf.beat_schedule = {
    'crawl-lotto-numbers': {
        'task': 'chatbot.tasks.update_lotto_data',
        'schedule': crontab(day_of_week='saturday', hour='22', minute='0'),
    },
    'train-ml-model': {
        'task': 'chatbot.tasks.train_ml_model',
        'schedule': crontab(day_of_week='saturday', hour='22', minute='30'),
    },
}

# 등록된 Django 앱 설정에서 task 불러오기
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
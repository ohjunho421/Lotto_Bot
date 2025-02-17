from celery import shared_task
import logging
from .services import LottoDataCollector, LottoPredictor
from django.conf import settings
import os

logger = logging.getLogger(__name__)

@shared_task
def update_lotto_data():
    """매주 토요일 22:00에 실행되는 크롤링 태스크"""
    try:
        collector = LottoDataCollector()
        success = collector.update_latest_data()
        
        if success:
            logger.info("로또 데이터 업데이트 완료")
        else:
            logger.warning("새로운 로또 데이터가 없거나 업데이트 실패")
            
        return success
    except Exception as e:
        logger.error(f"로또 데이터 업데이트 실패: {str(e)}")
        return False

@shared_task
def train_ml_model():
    """매주 토요일 22:30에 실행되는 모델 학습 태스크"""
    try:
        # 데이터 파일 존재 확인
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.error("로또 데이터 파일이 없습니다.")
            return False

        predictor = LottoPredictor()
        predictor.train_model()
        logger.info("머신러닝 모델 학습 완료")
        return True
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {str(e)}")
        return False

@shared_task
def initialize_data():
    """초기 데이터 수집 태스크"""
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            collector = LottoDataCollector()
            df = collector.collect_initial_data()
            if df is not None:
                logger.info("초기 데이터 수집 완료")
                return True
            else:
                logger.error("초기 데이터 수집 실패")
                return False
        else:
            logger.info("데이터 파일이 이미 존재합니다.")
            return True
            
    except Exception as e:
        logger.error(f"초기 데이터 수집 실패: {str(e)}")
        return False
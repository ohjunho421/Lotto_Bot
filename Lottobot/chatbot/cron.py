# chatbot/cron.py
def update_lotto_draws():
    from .services import crawl_lotto_data, train_lotto_model
    try:
        crawl_lotto_data()
        train_lotto_model()
        print("크롤링 및 모델 재학습 완료.")
    except Exception as e:
        print("크론 작업 오류:", str(e))
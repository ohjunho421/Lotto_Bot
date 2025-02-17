# chatbot/lotto_ml.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from django.conf import settings

def convert_draw_to_binary(row):
    """
    한 회차의 당첨번호(컬럼 '1','2','3','4','5','6')를 45차원의 이진 벡터로 변환.
    당첨된 번호면 해당 인덱스(0~44)에 1, 아니면 0.
    """
    binary = np.zeros(45)
    for col in ['1','2','3','4','5','6']:
        try:
            num = int(row[col])
            binary[num-1] = 1
        except:
            pass
    return binary

def create_features_targets(df, window_size=5):
    """
    슬라이딩 윈도우로 feature와 target을 생성합니다.
    - feature: 최근 window_size회차의 이진 벡터들을 flatten하고, 추가 통계 피처(전역, 최근3년, 최근1년 빈도)를 붙임
    - target: 다음 회차의 당첨 결과를 45차원 이진 벡터로 표현
    """
    features = []
    targets = []
    
    # 각 회차를 이진 벡터로 변환
    binary_draws = df.apply(convert_draw_to_binary, axis=1).tolist()
    binary_draws = np.array(binary_draws)
    
    # 전역 빈도 (전체 10년치, 정규화된 빈도)
    global_freq = np.sum(binary_draws, axis=0) / len(binary_draws)
    
    # 최근 빈도 계산 (간단히 마지막 10%를 최근1년, 30%를 최근3년으로 가정)
    n = len(binary_draws)
    recent1_freq = np.sum(binary_draws[int(n*0.9):], axis=0) / (n - int(n*0.9))
    recent3_freq = np.sum(binary_draws[int(n*0.7):], axis=0) / (n - int(n*0.7))
    
    for i in range(window_size, len(binary_draws) - 1):
        # 최근 window_size 회차의 이진 벡터를 flatten
        window_feat = binary_draws[i-window_size:i].flatten()
        
        # 추가 통계 피처: 전역, 최근3년, 최근1년 빈도 (각 45차원)
        feat = np.concatenate([window_feat, global_freq, recent3_freq, recent1_freq])
        features.append(feat)
        
        # 타겟: i번째 회차 당첨 결과 (45차원)
        targets.append(binary_draws[i])
    
    X = np.array(features)
    y = np.array(targets)
    return X, y

def train_lotto_model(window_size=5):
    """
    CSV 파일에서 데이터를 로드하고, 머신러닝 모델을 학습합니다.
    학습된 모델, scaler, 그리고 사용된 DataFrame(df)를 반환합니다.
    """
    # CSV 파일 경로는 settings에서 가져옴
    data_file = settings.LOTTO_DATA_FILE
    df = pd.read_csv(data_file)
    df = df.sort_values('회차').reset_index(drop=True)
    
    X, y = create_features_targets(df, window_size=window_size)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 학습/테스트 분리 (테스트는 확인용)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    base_clf = LogisticRegression(max_iter=1000)
    multi_clf = MultiOutputClassifier(base_clf)
    multi_clf.fit(X_train, y_train)
    
    return multi_clf, scaler, df

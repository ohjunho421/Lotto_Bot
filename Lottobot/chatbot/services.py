import numpy as np

def get_recommendation(strategy):
    """
    전략에 맞는 번호를 추천합니다.
    전략 1: 평균보다 많이 나온 번호
    전략 2: 평균보다 적게 나온 번호
    """
    # 실제 데이터 기반 로직으로 대체할 수 있습니다.
    probs = np.random.rand(45)  # 1~45 사이의 확률값을 임의로 생성 (기존 머신러닝 데이터를 기반으로 수정 가능)
    mean_val = np.mean(probs)
    std_val = np.std(probs)

    if strategy == 1:
        # 평균 + 표준편차 이상인 번호 추천 (많이 나온 번호)
        return sorted([i + 1 for i, prob in enumerate(probs) if prob >= mean_val + std_val][:6])
    elif strategy == 2:
        # 평균 - 표준편차 ~ 평균 사이의 번호 추천 (적게 나온 번호)
        return sorted([i + 1 for i, prob in enumerate(probs) if mean_val - std_val <= prob < mean_val][:6])
    else:
        raise ValueError("Invalid strategy")

# services.py

import os
import logging
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from joblib import dump, load

logger = logging.getLogger(__name__)

class LottoDataCollector:
    def __init__(self):
        self.base_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin"
        self.data_file = settings.LOTTO_DATA_FILE

    def collect_initial_data(self):
        """초기 데이터 수집"""
        try:
            logger.info("초기 데이터 수집 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.error(f"HTTP 오류: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 당첨번호 추출
            win_numbers = soup.select('div.num.win span.ball_645')
            if not win_numbers or len(win_numbers) != 6:
                logger.error("당첨번호를 찾을 수 없습니다")
                return None

            # 보너스 번호 추출
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            if not bonus_ball or len(bonus_ball) != 1:
                logger.error("보너스 번호를 찾을 수 없습니다")
                return None

            # 회차 정보 추출
            draw_result = soup.select('div.win_result h4')
            if not draw_result:
                logger.error("회차 정보를 찾을 수 없습니다")
                return None

            # 추첨일 추출
            draw_date = soup.select('p.desc')
            if not draw_date:
                logger.error("추첨일을 찾을 수 없습니다")
                return None

            try:
                # 데이터 파싱
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                drawn_date = date_text[date_text.find('(')+1:date_text.find(')')]

                logger.info(f"추출된 데이터: 회차={draw_no}, 번호={numbers}, 보너스={bonus}, 날짜={drawn_date}")

                # 데이터프레임 생성
                df = pd.DataFrame([{
                    '회차': draw_no,
                    '추첨일': drawn_date,
                    '1': numbers[0],
                    '2': numbers[1],
                    '3': numbers[2],
                    '4': numbers[3],
                    '5': numbers[4],
                    '6': numbers[5],
                    '보너스': bonus
                }])

                # 데이터 저장
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                df.to_csv(self.data_file, index=False)
                logger.info(f"데이터 파일 저장 완료: {self.data_file}")
                return df

            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"초기 데이터 수집 중 오류 발생: {str(e)}")
            return None

            if all_data:
                df = pd.DataFrame(all_data)
                df = df.sort_values('회차', ascending=False)
                logger.info(f"데이터프레임 생성 완료. 총 {len(df)}개의 데이터")
                
                # 데이터 저장 전에 디렉토리 확인
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                df.to_csv(self.data_file, index=False)
                logger.info(f"데이터 파일 저장 완료: {self.data_file}")
                return df
            else:
                logger.error("수집된 데이터가 없습니다.")
                return None

        except Exception as e:
            logger.error(f"초기 데이터 수집 중 오류 발생: {str(e)}")
            return None

    # services.py 파일 내 LottoDataCollector 클래스의 메서드들

    def _parse_date(self, date_text):
        """크롤링한 날짜를 YYYY.MM.DD 형식으로 변환"""
        try:
            # '2025년 02월 15일' 형식에서 숫자만 추출
            date_parts = ''.join(filter(str.isdigit, date_text))
            year = date_parts[:4]
            month = date_parts[4:6]
            day = date_parts[6:8]
            return f"{year}.{month}.{day}"
        except Exception as e:
            logger.error(f"날짜 파싱 오류: {str(e)}")
            return date_text

    def update_latest_data(self):
        """최신 데이터 업데이트 (CSV 형식을 첫번째 파일과 동일하게 통일)"""
        try:
            logger.info("최신 데이터 업데이트 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.error(f"HTTP 오류: {response.status_code}")
                return False

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 당첨번호 추출 (ball_645 클래스 사용)
            win_numbers = soup.select('div.num.win span.ball_645')
            if not win_numbers or len(win_numbers) != 6:
                logger.error("당첨번호를 찾을 수 없습니다")
                return False

            # 보너스 번호 추출
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            if not bonus_ball or len(bonus_ball) != 1:
                logger.error("보너스 번호를 찾을 수 없습니다")
                return False

            # 회차 정보 추출
            draw_result = soup.select('div.win_result h4')
            if not draw_result:
                logger.error("회차 정보를 찾을 수 없습니다")
                return False

            # 추첨일 추출
            draw_date = soup.select('p.desc')
            if not draw_date:
                logger.error("추첨일을 찾을 수 없습니다")
                return False

            try:
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                # 괄호 안의 날짜를 추출하여 전처리
                draw_date_extracted = date_text[date_text.find('(')+1:date_text.find(')')]
                draw_date_formatted = self._parse_date(draw_date_extracted)

                logger.info(f"추출된 데이터: 회차={draw_no}, 번호={numbers}, 보너스={bonus}, 날짜={draw_date_formatted}")
            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return False

            # 기존 CSV 파일 로드 (없으면 모든 컬럼을 포함한 빈 DataFrame 생성)
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
            else:
                df = pd.DataFrame(columns=[
                    '회차', '추첨일', '1', '2', '3', '4', '5', '6', '보너스',
                    '날짜', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6'
                ])

            # 신규 회차가 존재하지 않을 때만 추가
            if draw_no not in df['회차'].values:
                new_row = pd.DataFrame([{
                    '회차': draw_no,
                    '추첨일': draw_date_formatted,
                    '1': numbers[0],
                    '2': numbers[1],
                    '3': numbers[2],
                    '4': numbers[3],
                    '5': numbers[4],
                    '6': numbers[5],
                    '보너스': bonus
                }])
                
                # 신규 데이터는 원본 형식(회차,추첨일,1~6,보너스)만 가지고 있음.
                # 기존 데이터와 병합
                updated_df = pd.concat([new_row, df], ignore_index=True)
                
                # **형식 통일 작업 시작**
                # 1. '추첨일'이 비어있는 행은 '날짜' 컬럼 값으로 채우기
                if '날짜' in updated_df.columns:
                    updated_df['추첨일'] = updated_df['추첨일'].fillna(updated_df['날짜'])
                
                # 2. 번호 컬럼(1~6)이 비어있는 경우 기존 '번호1'~'번호6' 컬럼 값을 채우기
                old_number_cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                new_number_cols = ['1', '2', '3', '4', '5', '6']
                for new_col, old_col in zip(new_number_cols, old_number_cols):
                    if old_col in updated_df.columns:
                        updated_df[new_col] = updated_df[new_col].fillna(updated_df[old_col])
                
                # 3. 최종적으로 필요한 컬럼만 선택: 회차, 추첨일, 1,2,3,4,5,6,보너스
                final_columns = ['회차', '추첨일'] + new_number_cols + ['보너스']
                updated_df = updated_df[final_columns]
                
                # 4. 번호 컬럼들을 숫자형으로 변환 (필요 시 소수점 제거)
                for col in new_number_cols + ['보너스']:
                    updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
                # **형식 통일 작업 끝**
                
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                updated_df.to_csv(self.data_file, index=False)
                logger.info(f"신규 회차 {draw_no} 추가 및 형식 재정렬 완료")
                return True
            
            logger.info(f"회차 {draw_no}는 이미 존재함")
            return False

        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
            return False



class LottoPredictor:
    def __init__(self):
        self.model = LogisticRegression(multi_class='ovr', max_iter=1000)
        self.scaler = StandardScaler()
        self.model_file = os.path.join(settings.BASE_DIR, 'data', 'lotto_model.pkl')
        self.scaler_file = os.path.join(settings.BASE_DIR, 'data', 'lotto_scaler.pkl')
        self.stats_file = os.path.join(settings.BASE_DIR, 'data', 'model_stats.json')

    def save_model(self):
        """학습된 모델 저장"""
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            dump(self.model, self.model_file)
            dump(self.scaler, self.scaler_file)
            
            # 모델 학습 통계 저장
            stats = {
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_size': len(self.recent_data),
                'feature_size': self.recent_features.shape[1] if hasattr(self, 'recent_features') else 0
            }
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f)
                
            logger.info("모델 저장 완료")
            return True
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
            return False

    def load_model(self):
        """저장된 모델 로드"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = load(self.model_file)
                self.scaler = load(self.scaler_file)
                logger.info("저장된 모델 로드 완료")
                return True
            return False
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return False

    def train_model(self):
        """모델 학습"""
        try:
            if not os.path.exists(settings.LOTTO_DATA_FILE):
                logger.error("데이터 파일이 존재하지 않습니다")
                return False

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            self.recent_data = df
            logger.info(f"데이터 로드 완료: {len(df)}개의 데이터")

            X = self.prepare_features(df)
            self.recent_features = X
            logger.info(f"특성 데이터 준비 완료: {X.shape}")

            y = []
            for i in range(len(df) - 5):
                next_number = df.iloc[i]['1']  # 다음 회차의 첫 번째 번호
                y.append(next_number)
            y = np.array(y)

            X = self.scaler.fit_transform(X)
            self.model.fit(X, y)
            
            # 학습 완료 후 모델 저장
            self.save_model()
            
            logger.info("모델 학습 및 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            return False

    def predict_probabilities(self):
        """각 번호의 출현 확률 예측"""
        try:
            # 저장된 모델이 없으면 새로 학습
            if not self.load_model():
                if not self.train_model():
                    return np.ones(45) / 45

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            latest_features = self.prepare_features(df)[-1:]
            latest_features = self.scaler.transform(latest_features)
            probs = self.model.predict_proba(latest_features)
            return np.mean(probs, axis=0)
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            return np.ones(45) / 45

def get_recommendation(strategy_counts):
    """전략별 로또 번호 추천"""
    try:
        # 데이터 파일 확인 및 로드
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.info("데이터 파일이 없습니다. 초기 데이터를 수집합니다.")
            collector = LottoDataCollector()
            collector.collect_initial_data()

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        predictor = LottoPredictor()
        predictor.train_model()
        
        # 번호별 출현 빈도 분석
        all_numbers = []
        for col in ['1', '2', '3', '4', '5', '6']:
            all_numbers.extend(df[col].tolist())
        number_counts = pd.Series(all_numbers).value_counts()
        
        # 평균과 표준편차 계산
        mean_freq = number_counts.mean()
        std_freq = number_counts.std() 
        
        # 머신러닝 예측 확률 가져오기
        predicted_probs = predictor.predict_probabilities()
        
        recommendations = []
        for strategy, count in strategy_counts.items():
            strategy = int(strategy)
            for _ in range(count):
                if strategy == 1:
                    # 전략 1: 평균 이상 출현하는 번호들
                    candidates = [n for n in range(1, 46) if number_counts.get(n, 0) >= mean_freq]
                else:
                    # 전략 2: 평균 ~ (평균-표준편차) 구간의 번호들
                    candidates = [n for n in range(1, 46) if mean_freq > number_counts.get(n, 0) >= (mean_freq - std_freq)]
                
                # 출현 빈도와 ML 예측 확률을 결합한 가중치 계산
                if strategy == 1:
                    freq_weights = [(number_counts.get(n, 0) - mean_freq) / (number_counts.max() - mean_freq) for n in candidates]
                else:
                    freq_weights = [(mean_freq - number_counts.get(n, 0)) / mean_freq for n in candidates]
                
                ml_weights = [predicted_probs[n-1] for n in candidates]
                
                # 두 가중치를 동일 비율로 결합 (0.5:0.5)
                weights = [0.5 * fw + 0.5 * mw for fw, mw in zip(freq_weights, ml_weights)]
                weights = np.array(weights) / np.sum(weights)  # 정규화

                # 구간별 번호 선택
                selected = []
                ranges = [(1,15), (16,30), (31,45)]
                for start, end in ranges:
                    range_candidates = [n for n in candidates if start <= n <= end]
                    if range_candidates:
                        range_weights = [weights[candidates.index(n)] for n in range_candidates]
                        range_weights = np.array(range_weights) / np.sum(range_weights)
                        n_select = min(2, len(range_candidates))
                        selected.extend(np.random.choice(range_candidates, n_select, replace=False, p=range_weights))

                # 남은 자리 채우기
                remaining = 6 - len(selected)
                if remaining > 0:
                    remaining_candidates = [n for n in candidates if n not in selected]
                    if remaining_candidates:
                        remaining_weights = [weights[candidates.index(n)] for n in remaining_candidates]
                        remaining_weights = np.array(remaining_weights) / np.sum(remaining_weights)
                        selected.extend(np.random.choice(remaining_candidates, remaining, replace=False, p=remaining_weights))
                
                recommendations.append((strategy, sorted(selected)))
        
        return recommendations, None
        
    except Exception as e:
        logger.error(f"Error in get_recommendation: {str(e)}")
        return None, str(e)
    
def check_data_status():
    """데이터 상태 확인"""
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.warning("Lotto data file not found")
            return False, "데이터 파일이 없습니다. 초기 데이터를 수집해야 합니다."

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        if len(df) == 0:
            logger.warning("Empty data file")
            return False, "데이터 파일이 비어있습니다."

        # 최신 데이터 확인
        latest_date = pd.to_datetime(df['추첨일'].iloc[0])
        current_date = pd.Timestamp.now()
        days_diff = (current_date - latest_date).days

        if days_diff > 7:
            logger.warning(f"Data might be outdated. Last update: {latest_date}")
            return True, f"마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}\n{days_diff}일 전에 업데이트되었습니다."
        
        return True, f"데이터가 최신 상태입니다.\n마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}"

    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}")
        return False, f"데이터 상태 확인 중 오류 발생: {str(e)}"
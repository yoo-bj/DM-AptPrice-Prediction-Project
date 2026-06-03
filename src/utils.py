import numpy as np
import requests
import time
from sklearn.metrics.pairwise import haversine_distances
from dotenv import load_dotenv
import os

load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")


# 공공데이터 기준 최근접 시설까지 거리 계산 (미터 단위)
def nearest_distance(lat, lon, target_df, lat_col="위도", lon_col="경도"):
    coords1 = np.radians([[lat, lon]])
    coords2 = np.radians(target_df[[lat_col, lon_col]].values)
    distances = haversine_distances(coords1, coords2) * 6371000
    return distances.min()


# 공공데이터 기준 반경 내 시설 개수 카운트 (미터 단위)
def count_within_radius(lat, lon, target_df, radius=1000, lat_col="위도", lon_col="경도"):
    coords1 = np.radians([[lat, lon]])
    coords2 = np.radians(target_df[[lat_col, lon_col]].values)
    distances = haversine_distances(coords1, coords2) * 6371000
    return (distances[0] <= radius).sum()


# 카카오 지오코딩 API - 도로명 주소를 위도/경도로 변환
def get_coords_kakao(address):
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    try:
        response = requests.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            headers=headers,
            params=params
        )
        result = response.json()
        # 주소 검색 실패시 키워드 검색으로 재시도
        if not result.get("documents"):
            response = requests.get(
                "https://dapi.kakao.com/v2/local/search/keyword.json",
                headers=headers,
                params=params
            )
            result = response.json()
        if result.get("documents"):
            x = float(result["documents"][0]["x"])
            y = float(result["documents"][0]["y"])
            return y, x
    except Exception as e:
        print(f"오류 발생: {address} -> {e}")
    return None, None
"""
변수 생성 함수 모음.

==========================================================================
팀 규약 (회의 합의 사항)
==========================================================================

각 팀원은 자기 담당 변수 그룹의 함수를 이 파일에 추가한다.
통합 담당자는 02_features.ipynb에서 이 함수들을 순서대로 호출하여
마스터 데이터셋(apt_master.csv)을 생성한다.

[함수 시그니처 규약]
    def add_{그룹}_features(df, ...) -> DataFrame:
        - 첫 번째 인자는 항상 df (실거래가 데이터)
        - 외부 데이터프레임을 인자로 받음
        - 변수 추가된 df를 반환
        - 함수 docstring에 추가하는 컬럼명 명시

[컬럼명 규약]
    - 거리 변수: {prefix}_nearest_dist       (예: subway_nearest_dist)
    - 카운트 변수: {prefix}_count_{radius}m  (예: mart_count_1000m)
    - 특수 변수는 자유롭게 (단, prefix는 통일)

[충돌 방지]
    - 각자 자기 그룹 섹션(아래 "===" 구분선) 안에서만 함수 추가/수정
    - 다른 사람 함수는 건드리지 않음
    - 공용 헬퍼(add_features, add_features_unique)는 통합 담당자만 수정
"""

import pandas as pd
from utils import nearest_distance, count_within_radius


# ==========================================================================
# 공용 헬퍼 함수 (통합 담당자만 수정)
# ==========================================================================

def add_features(df, target_df, prefix, radius=1000,
                 lat_col='위도', lon_col='경도'):
    """
    좌표 기반 변수 2개를 한 번에 추가하는 템플릿 함수.

    Parameters
    ----------
    df : DataFrame
        실거래가 데이터 (위도/경도 컬럼 보유)
    target_df : DataFrame
        외부 시설 데이터 (위도/경도 컬럼 보유)
    prefix : str
        새 컬럼명 접두사. 예: 'subway', 'mart', 'academy'
    radius : int
        카운트 반경 (미터)
    lat_col, lon_col : str
        target_df 안에서 위도/경도 컬럼명 (한글 외 다른 이름이면 명시)

    Returns
    -------
    DataFrame
        다음 두 컬럼이 추가된 df:
        - {prefix}_nearest_dist   : 최근접 시설까지 거리(m)
        - {prefix}_count_{radius}m : 반경 내 시설 개수

    Examples
    --------
    >>> df = add_features(df, subway, 'subway', radius=500)
    >>> df = add_features(df, mart, 'mart', radius=1000, lat_col='LAT', lon_col='LON')
    """
    dist_col = f'{prefix}_nearest_dist'
    count_col = f'{prefix}_count_{radius}m'

    df[dist_col] = df.apply(
        lambda row: nearest_distance(
            row['위도'], row['경도'], target_df, lat_col=lat_col, lon_col=lon_col
        ),
        axis=1
    )
    df[count_col] = df.apply(
        lambda row: count_within_radius(
            row['위도'], row['경도'], target_df, radius=radius,
            lat_col=lat_col, lon_col=lon_col
        ),
        axis=1
    )

    print(f'[{prefix}] 변수 추가 완료: {dist_col}, {count_col}')
    return df


def add_features_unique(df, target_df, prefix, radius=1000,
                        key_cols=['단지명', '도로명'],
                        lat_col='위도', lon_col='경도'):
    """
    add_features의 최적화 버전. 단지별로 중복 제거 후 계산하여 속도 개선.
    실거래가는 같은 단지가 수십~수백 번 등장하므로 중복 계산 방지에 효과적.

    Parameters
    ----------
    df, target_df, prefix, radius, lat_col, lon_col : add_features와 동일
    key_cols : list of str
        단지 식별 컬럼 (기본: ['단지명', '도로명'])

    Examples
    --------
    >>> df = add_features_unique(df, subway, 'subway', radius=500)
    """
    dist_col = f'{prefix}_nearest_dist'
    count_col = f'{prefix}_count_{radius}m'

    # 단지별 중복 제거
    unique_apt = df[key_cols + ['위도', '경도']].drop_duplicates().reset_index(drop=True)

    unique_apt[dist_col] = unique_apt.apply(
        lambda row: nearest_distance(
            row['위도'], row['경도'], target_df, lat_col=lat_col, lon_col=lon_col
        ),
        axis=1
    )
    unique_apt[count_col] = unique_apt.apply(
        lambda row: count_within_radius(
            row['위도'], row['경도'], target_df, radius=radius,
            lat_col=lat_col, lon_col=lon_col
        ),
        axis=1
    )

    # 원본 데이터에 병합
    df = df.merge(unique_apt[key_cols + [dist_col, count_col]],
                  on=key_cols, how='left')

    print(f'[{prefix}] 변수 추가 완료 (고유 단지 {len(unique_apt)}개 기준): '
          f'{dist_col}, {count_col}')
    return df


# ==========================================================================
# 교통 변수 — 담당: A
# ==========================================================================

def add_transport_features(df, subway_df=None, bus_df=None,
                           subway_radius=500, bus_radius=500):
    """
    교통 접근성 변수 추가.

    추가 컬럼:
        - subway_nearest_dist     : 최근접 지하철역 거리(m)
        - subway_count_{r}m       : 반경 내 지하철역 개수
        - bus_count_{r}m          : 반경 내 버스정류장 개수

    Parameters
    ----------
    df : DataFrame
        실거래가 데이터
    subway_df : DataFrame, optional
        지하철역 데이터 (None이면 스킵)
    bus_df : DataFrame, optional
        버스정류장 데이터 (None이면 스킵)
    """
    if subway_df is not None:
        df = add_features_unique(df, subway_df, 'subway', radius=subway_radius)
    if bus_df is not None:
        df = add_features_unique(df, bus_df, 'bus', radius=bus_radius)
    return df


# ==========================================================================
# 교육 변수 — 담당: B
# ==========================================================================

def add_education_features(df, school_df=None, academy_df=None,
                           school_radius=1000, academy_radius=500):
    """
    교육 환경 변수 추가.

    추가 컬럼 (예시 — 담당자가 자유롭게 조정):
        - school_nearest_dist
        - school_count_{r}m
        - academy_count_{r}m

    TODO: 담당자가 실제 데이터 컬럼 구조 확인 후 구현
    """
    if school_df is not None:
        df = add_features_unique(df, school_df, 'school', radius=school_radius)
    if academy_df is not None:
        df = add_features_unique(df, academy_df, 'academy', radius=academy_radius)
    return df


# ==========================================================================
# 생활편의 변수 — 담당: C
# ==========================================================================

def add_convenience_features(df, mart_df=None, hospital_df=None,
                             mart_radius=1000, hospital_radius=1500):
    """
    생활편의 변수 추가.

    추가 컬럼 (예시):
        - mart_nearest_dist
        - hospital_nearest_dist

    TODO: 담당자가 실제 데이터 컬럼 구조 확인 후 구현
    """
    if mart_df is not None:
        df = add_features_unique(df, mart_df, 'mart', radius=mart_radius)
    if hospital_df is not None:
        df = add_features_unique(df, hospital_df, 'hospital', radius=hospital_radius)
    return df


# ==========================================================================
# 환경(부정) 변수 — 담당: D
# ==========================================================================

def add_negative_features(df, funeral_df=None, incinerator_df=None,
                          funeral_radius=2000, incinerator_radius=3000):
    """
    환경 부정 요소 변수 추가.

    추가 컬럼 (예시):
        - funeral_nearest_dist
        - incinerator_nearest_dist

    TODO: 담당자가 실제 데이터 컬럼 구조 확인 후 구현
    """
    if funeral_df is not None:
        df = add_features_unique(df, funeral_df, 'funeral', radius=funeral_radius)
    if incinerator_df is not None:
        df = add_features_unique(df, incinerator_df, 'incinerator', radius=incinerator_radius)
    return df


# ==========================================================================
# 특수 변수 — 표준 패턴(거리/카운트)이 안 맞는 경우 여기에 추가
# ==========================================================================

# 예: 학교 종류별 카운트, 특정 노선 지하철 거리 등
# def add_school_by_level(df, school_df):
#     """초/중/고 종류별 카운트 같은 특수 변수"""
#     ...
#     return df

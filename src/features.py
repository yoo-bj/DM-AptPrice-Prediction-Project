"""
변수 생성 헬퍼 함수 모음.

기본 사용 패턴:
    df = add_features(df, target_df=mart_df, prefix='mart', radius=1000)
    # → df['mart_nearest_dist'], df['mart_count_1000m'] 컬럼 추가

특수 변수가 필요하면 이 파일에 자기 함수를 추가해서 사용.
"""

import pandas as pd
from utils import nearest_distance, count_within_radius


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

    사용 시점: 변수 계산이 오래 걸리는 경우 (예: 외부 데이터가 큰 경우)

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
# 특수 변수가 필요하면 아래에 자기 함수를 추가
# ==========================================================================

# def add_school_by_level(df, school_df, ...):
#     """초/중/고 종류별 카운트 같은 특수 변수 함수 예시"""
#     ...
"""
신호처리 실습 유틸리티 — 단일 파일

실습 노트북 [a_Practice_Python], [b_basics], [c_advanced] 전부가 이 모듈 하나만 참조한다.
교재 본문 작업과는 분리되어 있으며, Colab 독립 실행을 염두에 둔다.

제공 함수
---------
fft(x, fs)                                  단측 진폭 스펙트럼
bandpass(x, fs, f_low, f_high, order=4)     Butterworth 대역통과 (filtfilt)
hilbert_envelope(x)                         힐베르트 포락선
resampling(t, v, deg, f_resampling, trig)   각도 기반 재샘플링 (order analysis)
MEDA(x, FilterSize=5, remain=False)         최소엔트로피 디콘볼루션
feature_time(v)                             RMS / Skew / Kurt / CF
feature_freq(v, fs, band)                   주파수 대역별 RMS
draw_features(...)                          feature 비교 그림
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# FFT (단측 진폭 스펙트럼)
# ---------------------------------------------------------------
def fft(x, fs, n=None):
    """rfft 기반 단측 진폭 스펙트럼.

    Parameters
    ----------
    x : array_like (1-D)
        시간 신호
    fs : float
        샘플링 주파수 (Hz)
    n : int, optional
        FFT 길이. 기본값은 `len(x)`.
        - n > len(x) : 신호 끝에 0 을 붙여 **제로패딩** 후 FFT
          → bin 간격이 `fs/n` 으로 촘촘해져 피크 위치를 더 정밀히 읽을 수 있음
        - n < len(x) : 신호를 `n` 샘플까지 잘라냄
        - n == None : 기본 (원 신호 그대로)
        진폭 정규화는 **실제 데이터 길이** `N = len(x)` 기준이다
        (제로패딩한 0 은 에너지가 아니므로 분모에 포함하지 않는다).

    Returns
    -------
    f : ndarray
        주파수 축 0 ~ fs/2, 길이 `n//2+1`
    A : ndarray
        단측 진폭 스펙트럼, 길이 `n//2+1`
    """
    x = np.asarray(x).flatten()
    N = len(x)
    if n is None:
        n = N
    Xr = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    A = 2.0 * np.abs(Xr) / N
    A[0] /= 2.0
    if n % 2 == 0:
        A[-1] /= 2.0
    return f, A


def fft_full(x, fs):
    """양측 스펙트럼 (fftshift 적용, 복소 반환).

    Parameters
    ----------
    x : array_like
    fs : float
        샘플링 주파수 (Hz)

    Returns
    -------
    f : ndarray (N,)
        주파수 축 -fs/2 ~ +fs/2 (fftshift 적용)
    X : ndarray (N,)
        복소 스펙트럼 (1/N 로 정규화 — 진폭 해석 용이)
    """
    x = np.asarray(x).flatten()
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x)) / N
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))
    return f, X


# ---------------------------------------------------------------
# 필터링 (Butterworth, zpk→sos→k 정규화 + sosfilt)
# ---------------------------------------------------------------
def filtering(x, fs, ftype='band', f_low=None, f_high=None, f_cut=None, order=4):
    """Butterworth 필터링.

    Nyquist 경계 근처에서도 수치적으로 안정하도록 `butter(..., output='zpk')` →
    `zpk2sos` → 첫 SOS 섹션에서 `k` 를 분리해 `sosfilt` 적용 후 다시 `*k` 로
    복원하는 방식으로 구현. `filtfilt` (zero-phase) 가 아닌 `sosfilt` (causal,
    single-pass) 이므로 위상 지연이 발생한다.

    Parameters
    ----------
    x : array_like
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    ftype : {'band', 'bandpass', 'high', 'highpass', 'low', 'lowpass'}
        필터 종류
    f_low, f_high : float, optional
        'band' 일 때 하한/상한 주파수 (Hz)
    f_cut : float, optional
        'high' 또는 'low' 일 때 차단 주파수 (Hz)
    order : int
        필터 차수 (기본 4)

    Returns
    -------
    y : ndarray
        필터링된 신호

    Examples
    --------
    >>> y = filtering(x, fs, 'band', f_low=5, f_high=12)
    >>> y = filtering(x, fs, 'high', f_cut=5)
    >>> y = filtering(x, fs, 'low',  f_cut=5)
    """
    nyq = fs / 2.0
    t = ftype.lower()
    if t in ('band', 'bandpass'):
        if f_low is None or f_high is None:
            raise ValueError("'band' 필터에는 f_low, f_high 를 지정해야 합니다.")
        Wn, btype = np.array([f_low, f_high]) / nyq, 'band'
    elif t in ('high', 'highpass'):
        if f_cut is None:
            raise ValueError("'high' 필터에는 f_cut 을 지정해야 합니다.")
        Wn, btype = f_cut / nyq, 'high'
    elif t in ('low', 'lowpass'):
        if f_cut is None:
            raise ValueError("'low' 필터에는 f_cut 을 지정해야 합니다.")
        Wn, btype = f_cut / nyq, 'low'
    else:
        raise ValueError(f"ftype 은 'band'|'high'|'low' 중 하나여야 합니다 (입력: {ftype!r})")

    z, p, k = signal.butter(order, Wn, btype=btype, output='zpk')
    sos = signal.zpk2sos(z, p, k)
    # 첫 SOS 섹션의 numerator 에 누적된 gain 을 빼내 수치 안정성 향상
    sos[0, 0:3] = sos[0, 0:3] / k
    return signal.sosfilt(sos, x) * k


def filtering_zerophase(x, fs, ftype='band', f_low=None, f_high=None, f_cut=None, order=4):
    """Butterworth 필터링 — **zero-phase** 버전.

    `filtering` 과 동일한 SOS 구현을 **`sosfiltfilt`** 로 전·후방 2회 적용하여
    위상 지연을 0 으로 만든다. 크기 응답은 $|H|^2$ 가 되므로 cutoff 에서 −6 dB
    (단일 `sosfilt`) 가 아니라 −12 dB 지점이 된다. 인과성(실시간 동작) 이
    필요 없는 **사후 분석용**.

    Parameters
    ----------
    동일. `filtering` 참고.

    Returns
    -------
    y : ndarray
        zero-phase 필터링된 신호 (입력과 길이·위상 동일)
    """
    nyq = fs / 2.0
    t = ftype.lower()
    if t in ('band', 'bandpass'):
        if f_low is None or f_high is None:
            raise ValueError("'band' 필터에는 f_low, f_high 를 지정해야 합니다.")
        Wn, btype = np.array([f_low, f_high]) / nyq, 'band'
    elif t in ('high', 'highpass'):
        if f_cut is None:
            raise ValueError("'high' 필터에는 f_cut 을 지정해야 합니다.")
        Wn, btype = f_cut / nyq, 'high'
    elif t in ('low', 'lowpass'):
        if f_cut is None:
            raise ValueError("'low' 필터에는 f_cut 을 지정해야 합니다.")
        Wn, btype = f_cut / nyq, 'low'
    else:
        raise ValueError(f"ftype 은 'band'|'high'|'low' 중 하나여야 합니다 (입력: {ftype!r})")

    z, p, k = signal.butter(order, Wn, btype=btype, output='zpk')
    sos = signal.zpk2sos(z, p, k)
    return signal.sosfiltfilt(sos, x)


# ---------------------------------------------------------------
# 힐베르트 포락선
# ---------------------------------------------------------------
def hilbert_envelope(x):
    """힐베르트 해석신호의 진폭 (포락선)."""
    return np.abs(hilbert(np.asarray(x)))


# ---------------------------------------------------------------
# Order analysis 재샘플링 (각도 기반)
# ---------------------------------------------------------------
def resampling(time, v_sample, degree_sample, f_resampling, trig_rot):
    """각도 축으로 등간격 재샘플링.

    Parameters
    ----------
    time : ndarray
        시간 축 (초)
    v_sample : ndarray
        신호 샘플
    degree_sample : ndarray
        각 샘플 시점의 회전 각도 (degree)
    f_resampling : int
        회전 1바퀴당 샘플 수
    trig_rot : int
        1이면 정수 바퀴만 유지 (뒤쪽 절단)

    Returns
    -------
    t_resampling, v_resampling, degree_resampling, f_resampling
    """
    idx_zero_deg = degree_sample == 0
    time = time[~idx_zero_deg]
    v_sample = v_sample[~idx_zero_deg]
    degree_sample = degree_sample[~idx_zero_deg]

    starting = degree_sample[0]
    ending = degree_sample[-1]
    degree_re_delta = 360.0 / f_resampling
    degree_resampling = np.arange(starting + degree_re_delta,
                                  ending - degree_re_delta,
                                  degree_re_delta)

    fx_t = interp1d(degree_sample, time)
    t_resampling = fx_t(degree_resampling)

    fx_v = interp1d(time, v_sample)
    v_resampling = fx_v(t_resampling)

    if trig_rot == 1:
        rem = len(v_resampling) % f_resampling
        if rem > 0:
            v_resampling = v_resampling[:-rem]

    return t_resampling, v_resampling, degree_resampling, f_resampling


# ---------------------------------------------------------------
# MED (Minimum Entropy Deconvolution) — 베어링 임펄스 강조
# ---------------------------------------------------------------
def MEDA(x, FilterSize=5, remain=False):
    """최소엔트로피 디콘볼루션 필터 계수와 필터링 결과 반환.

    Parameters
    ----------
    x : ndarray
        입력 신호
    FilterSize : int
        필터 길이
    remain : bool
        True면 원 신호 길이 유지

    Returns
    -------
    f_MED : ndarray
        최적 필터 계수
    y_MED : ndarray
        MED 필터링된 신호
    """
    N = len(x)
    X0 = np.zeros([FilterSize, N + FilterSize - 1])

    for n in range(FilterSize):
        if -FilterSize + n + 1 != 0:
            X0[n, n:-FilterSize + n + 1] = x
        else:
            X0[n, n:] = x

    if remain:
        X0 = X0[:, 0:N]
    else:
        X0 = X0[:, FilterSize:N]

    autocorr = X0 @ X0.T
    autocorr_inv = np.linalg.pinv(autocorr)
    F = autocorr_inv @ X0

    K = X0 * F
    dnorm = np.sum(K, axis=0)
    idx = np.argmax(dnorm)

    f_MED = F[:, idx]
    y_MED = X0.T @ f_MED

    return f_MED, y_MED


# ---------------------------------------------------------------
# 시간 영역 특징 (RMS, Skew, Kurt, CF)
# ---------------------------------------------------------------
def feature_time(v):
    """시간 영역 통계 특징.

    Returns
    -------
    features : ndarray [RMS, Skew, Kurt, CF]
    names    : list of str
    """
    v = np.asarray(v)
    RMS = np.sqrt(np.mean(v ** 2))
    SKEW = stats.skew(v)
    KURT = stats.kurtosis(v, axis=0, fisher=False)
    CF = np.max(v) / RMS

    features = np.array([RMS, SKEW, KURT, CF])
    names = ['RMS', 'Skew', 'Kurt', 'CF']
    return features, names


# ---------------------------------------------------------------
# 주파수 대역별 RMS
# ---------------------------------------------------------------
def feature_freq(v, fs, band=None):
    """주파수 대역별 진폭 RMS.

    Parameters
    ----------
    v : ndarray
        입력 신호
    fs : float
        샘플링 주파수
    band : array_like, shape (n_bands, 2)
        각 행이 [f_low, f_high] Hz

    Returns
    -------
    features : ndarray
    names    : list of str
    """
    features = []
    names = []

    if band is None:
        return np.array(features), names

    band = np.asarray(band)
    if band.size == 0:
        return np.array(features), names

    f, A = fft(v - np.mean(v), fs)

    n_bands = band.shape[0] if band.ndim == 2 else 1
    band = band.reshape(n_bands, 2)

    for n in range(n_bands):
        f_lo, f_hi = band[n, 0], band[n, 1]
        mask = (f > f_lo) & (f < f_hi)
        A_band = A[mask]
        if A_band.size > 0:
            features.append(np.sqrt(np.mean(A_band ** 2)))
        else:
            features.append(0.0)
        names.append(f'Band{n+1}')

    return np.array(features), names


# ---------------------------------------------------------------
# 종합 Feature (시간/주파수 + 필터링 + 포락선 대역 에너지)
# ---------------------------------------------------------------
def feature(v, fs, band_energy_and_filter, band_from_envelope):
    """종합 feature 벡터 추출 (원본 시간/주파수 + 필터링 시간/포락선 주파수).

    Parameters
    ----------
    v : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    band_energy_and_filter : list or tuple
        [f_low, f_high] — 원본 신호 대역 에너지 계산 + 밴드패스 필터 영역
    band_from_envelope : array_like, shape (n_bands, 2)
        포락선 스펙트럼에서 에너지 계산할 대역들

    Returns
    -------
    features : ndarray  (시간 + 원본 대역 + 필터링 시간 + 포락선 대역)
    names    : ndarray of str
    """
    f_low, f_high = band_energy_and_filter
    v_filter = filtering(v, fs, 'band', f_low=f_low, f_high=f_high)
    v_filter = np.asarray(v_filter).copy()
    v_filter[:100] = 0
    v_filter_env = hilbert_envelope(v_filter)

    feat_t,        name_t         = feature_time(v)
    feat_t_filter, _              = feature_time(v_filter)
    name_t_filter = [s + '_filter' for s in name_t]

    band_raw = np.asarray(band_energy_and_filter).reshape(1, 2)
    feat_f_raw, name_f_raw = feature_freq(v, fs, band=band_raw)

    feat_f_env, name_f_env = feature_freq(v_filter_env, fs, band=band_from_envelope)
    name_f_env = [s + '_env' for s in name_f_env]

    features = np.concatenate((feat_t, feat_f_raw, feat_t_filter, feat_f_env))
    names    = np.concatenate((name_t, name_f_raw, name_t_filter, name_f_env))
    return features, names


# ---------------------------------------------------------------
# Feature 비교 그림 (정상 vs 고장)
# ---------------------------------------------------------------
def draw_features(feature_t_n, feature_t_f, feature_t_name,
                  feature_f_n, feature_f_f, feature_f_name):
    """정상/고장 특징 비교 (절대값 + 비율)."""
    feature_t_ratio = feature_t_f / feature_t_n
    feature_f_ratio = feature_f_f / feature_f_n

    # 절대값
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(feature_t_n, '-o', color='C0', label='Normal')
    ax1.plot(feature_t_f, '-x', color='C1', label='Fault')
    ax1.set_xticks(np.arange(len(feature_t_name)))
    ax1.set_xticklabels(feature_t_name)
    ax1.set_ylabel('Time Features')
    ax1.legend()

    ax2.plot(feature_f_n, '-o', color='C0', label='Normal')
    ax2.plot(feature_f_f, '-x', color='C1', label='Fault')
    ax2.set_xticks(np.arange(len(feature_f_name)))
    ax2.set_xticklabels(feature_f_name)
    ax2.set_ylabel('Freq. Features')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.legend()
    fig.tight_layout()
    plt.show()

    # 비율
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(feature_t_ratio, '-o', color='C0')
    ax1.axhline(y=1, color='C3', linestyle='--')
    ax1.set_xticks(np.arange(len(feature_t_name)))
    ax1.set_xticklabels(feature_t_name)
    ax1.set_ylabel('Time Features Ratio')
    ax1.set_ylim(bottom=0)

    ax2.plot(feature_f_ratio, '-o', color='C0')
    ax2.axhline(y=1, color='C3', linestyle='--')
    ax2.set_xticks(np.arange(len(feature_f_name)))
    ax2.set_xticklabels(feature_f_name)
    ax2.set_ylabel('Freq. Features Ratio')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    plt.show()

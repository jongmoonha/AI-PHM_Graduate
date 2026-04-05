"""
신호처리 실습 유틸리티 — 필터링
"""

import numpy as np


def lowpass(x, fs, cutoff, order=4):
    """Butterworth 저역통과 필터.

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    cutoff : float
        차단 주파수 (Hz)
    order : int
        필터 차수 (기본 4)

    Returns
    -------
    y : ndarray
        필터링된 신호
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    y = filtfilt(b, a, x)
    return y


def bandpass(x, fs, f_low, f_high, order=4):
    """Butterworth 대역통과 필터.

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    f_low : float
        하한 주파수 (Hz)
    f_high : float
        상한 주파수 (Hz)
    order : int
        필터 차수 (기본 4)

    Returns
    -------
    y : ndarray
        필터링된 신호
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')
    y = filtfilt(b, a, x)
    return y


def fft_filter(x, fs, cutoff, filter_type='lowpass'):
    """FFT 기반 이상적 주파수 필터링.

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    cutoff : float or tuple
        차단 주파수. bandpass일 경우 (f_low, f_high) 튜플
    filter_type : str
        'lowpass', 'highpass', 'bandpass'

    Returns
    -------
    y : ndarray
        필터링된 신호
    """
    N = len(x)
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, d=1/fs)

    if filter_type == 'lowpass':
        mask = np.abs(freq) <= cutoff
    elif filter_type == 'highpass':
        mask = np.abs(freq) >= cutoff
    elif filter_type == 'bandpass':
        f_low, f_high = cutoff
        mask = (np.abs(freq) >= f_low) & (np.abs(freq) <= f_high)
    else:
        raise ValueError(f"지원하지 않는 filter_type: {filter_type}")

    X_filtered = X * mask
    y = np.real(np.fft.ifft(X_filtered))
    return y


def moving_average(x, window_size):
    """이동평균 필터.

    Parameters
    ----------
    x : ndarray
        입력 신호
    window_size : int
        윈도우 크기

    Returns
    -------
    y : ndarray
        필터링된 신호
    """
    kernel = np.ones(window_size) / window_size
    y = np.convolve(x, kernel, mode='same')
    return y


def med_filter(x, filter_length=50, n_iterations=30):
    """MED (Minimum Entropy Deconvolution) 필터.

    Parameters
    ----------
    x : ndarray
        입력 신호
    filter_length : int
        필터 길이 (기본 50)
    n_iterations : int
        반복 횟수 (기본 30)

    Returns
    -------
    y : ndarray
        필터링된 신호
    """
    from scipy.linalg import toeplitz

    N = len(x)
    L = filter_length

    # Toeplitz 행렬 구성
    col = np.zeros(N)
    col[:L] = x[:L]
    row = np.zeros(L)
    row[0] = x[0]
    X_mat = toeplitz(x, np.zeros(L))

    # 초기 필터 계수
    f = np.zeros(L)
    f[L // 2] = 1.0

    for _ in range(n_iterations):
        y = X_mat @ f
        # 커토시스 최대화
        y4 = y**3  # d(kurtosis)/dy ∝ y^3
        norm = np.sum(y**4)
        if norm < 1e-12:
            break
        f_new = X_mat.T @ y4 / norm
        # 정규화
        f_new = f_new / (np.linalg.norm(f_new) + 1e-12)
        f = f_new

    y = X_mat @ f
    return y

"""
신호처리 실습 유틸리티 — FFT 및 스펙트럼 분석
"""

import numpy as np


def fft(x, fs):
    """FFT 수행, 단측 진폭 스펙트럼 반환.

    Parameters
    ----------
    x : array_like
        시간 신호 (1-D)
    fs : float
        샘플링 주파수 (Hz)

    Returns
    -------
    amp : ndarray
        단측 진폭 스펙트럼 (N//2+1,)
    freq : ndarray
        주파수 축 (N//2+1,), 0 ~ fs/2
    """
    N = len(x)
    X = np.fft.fft(x)
    freq = np.fft.rfftfreq(N, d=1/fs)
    amp = 2.0 * np.abs(X[:N//2+1]) / N
    amp[0] /= 2  # DC 성분은 2배 불필요
    return amp, freq


def fft_full(x, fs):
    """FFT 수행, 양측(two-sided) 복소 스펙트럼 반환 (fftshift 적용).

    Returns
    -------
    X : ndarray
        복소 스펙트럼 (N,) — fftshift 적용
    freq : ndarray
        양측 주파수 축 (N,) — -fs/2 ~ fs/2
    """
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    return X, freq


def ifft_real(X, N=None):
    """IFFT 수행, 실수 신호 복원.

    Parameters
    ----------
    X : array_like
        복소 스펙트럼 (fft 출력, shift 안 된 상태)
    N : int, optional
        출력 길이 (기본: len(X))

    Returns
    -------
    x : ndarray
        복원된 실수 신호
    """
    return np.real(np.fft.ifft(X, n=N))


def to_db(x, ref=1.0):
    """진폭 → dB 변환 (20 * log10).

    Parameters
    ----------
    x : array_like
        진폭 배열
    ref : float
        기준값 (기본 1.0)

    Returns
    -------
    ndarray
        dB 값 배열
    """
    return 20 * np.log10(np.maximum(np.abs(x) / ref, 1e-12))


def psd_welch(x, fs, nperseg=256, **kwargs):
    """Welch PSD 추정 — scipy.signal.welch 래퍼.

    Returns
    -------
    freq : ndarray
        주파수 축
    psd : ndarray
        파워 스펙트럼 밀도
    """
    from scipy.signal import welch
    freq, psd = welch(x, fs=fs, nperseg=nperseg, **kwargs)
    return freq, psd

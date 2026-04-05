"""
신호처리 실습 유틸리티 — 포락선 분석 (힐베르트 변환, 켑스트럼)
"""

import numpy as np


def hilbert_envelope(x):
    """힐베르트 변환 기반 포락선(순시 진폭) 추출.

    Parameters
    ----------
    x : ndarray
        입력 신호

    Returns
    -------
    env : ndarray
        포락선 (순시 진폭)
    """
    from scipy.signal import hilbert
    analytic = hilbert(x)
    env = np.abs(analytic)
    return env


def envelope_spectrum(x, fs):
    """포락선 스펙트럼 계산 (힐베르트 포락선 → FFT).

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)

    Returns
    -------
    freq : ndarray
        주파수 축 (단측)
    amp : ndarray
        포락선 스펙트럼 진폭
    """
    env = hilbert_envelope(x)
    env = env - np.mean(env)  # DC 제거

    N = len(env)
    X = np.fft.fft(env)
    freq = np.fft.rfftfreq(N, d=1/fs)
    amp = 2.0 * np.abs(X[:N//2+1]) / N
    amp[0] /= 2
    return freq, amp


def cepstrum(x):
    """실수 Cepstrum 계산.

    Parameters
    ----------
    x : ndarray
        입력 신호

    Returns
    -------
    ceps : ndarray
        켑스트럼
    quefrency : ndarray
        쿼프런시 축 (샘플 단위)
    """
    X = np.fft.fft(x)
    log_X = np.log(np.maximum(np.abs(X), 1e-12))
    ceps = np.real(np.fft.ifft(log_X))
    quefrency = np.arange(len(ceps))
    return ceps, quefrency

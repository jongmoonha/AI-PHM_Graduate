"""
신호처리 실습 유틸리티 — 신호 생성
"""

import numpy as np


def sine(f, fs, duration, amplitude=1.0, phase=0.0):
    """정현파(sin) 신호 생성.

    Parameters
    ----------
    f : float
        주파수 (Hz)
    fs : float
        샘플링 주파수 (Hz)
    duration : float
        신호 길이 (초)
    amplitude : float
        진폭 (기본 1.0)
    phase : float
        초기 위상 (rad, 기본 0.0)

    Returns
    -------
    t : ndarray
        시간 축
    x : ndarray
        정현파 신호
    """
    t = np.arange(0, duration, 1/fs)
    x = amplitude * np.sin(2 * np.pi * f * t + phase)
    return t, x


def cosine(f, fs, duration, amplitude=1.0, phase=0.0):
    """코사인파 신호 생성.

    Returns
    -------
    t : ndarray
        시간 축
    x : ndarray
        코사인 신호
    """
    t = np.arange(0, duration, 1/fs)
    x = amplitude * np.cos(2 * np.pi * f * t + phase)
    return t, x


def composite_signal(freqs, amps, fs, duration):
    """다중 주파수 합성 신호 생성.

    Parameters
    ----------
    freqs : list of float
        주파수 리스트 [f1, f2, ...]
    amps : list of float
        진폭 리스트 [a1, a2, ...]
    fs : float
        샘플링 주파수 (Hz)
    duration : float
        신호 길이 (초)

    Returns
    -------
    t : ndarray
        시간 축
    x : ndarray
        합성 신호
    """
    t = np.arange(0, duration, 1/fs)
    x = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return t, x


def square_wave_fourier(t, f0, N_harmonics):
    """사각파의 푸리에 급수 부분합 계산.

    Parameters
    ----------
    t : ndarray
        시간 축
    f0 : float
        기본 주파수 (Hz)
    N_harmonics : int
        포함할 고조파 수

    Returns
    -------
    x : ndarray
        부분합 신호
    """
    x = np.zeros_like(t)
    for k in range(1, N_harmonics + 1, 2):  # 홀수 고조파만
        x += (4 / (np.pi * k)) * np.sin(2 * np.pi * k * f0 * t)
    return x


def chirp(f_start, f_end, fs, duration):
    """선형 chirp 신호 생성.

    Parameters
    ----------
    f_start : float
        시작 주파수 (Hz)
    f_end : float
        종료 주파수 (Hz)
    fs : float
        샘플링 주파수 (Hz)
    duration : float
        신호 길이 (초)

    Returns
    -------
    t : ndarray
        시간 축
    x : ndarray
        chirp 신호
    """
    from scipy.signal import chirp as scipy_chirp
    t = np.arange(0, duration, 1/fs)
    x = scipy_chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    return t, x


def add_noise(x, snr_db=None, noise_std=None, seed=None):
    """백색 가우시안 노이즈 추가.

    Parameters
    ----------
    x : ndarray
        원 신호
    snr_db : float, optional
        목표 SNR (dB) — noise_std와 택 1
    noise_std : float, optional
        노이즈 표준편차 — snr_db와 택 1
    seed : int, optional
        랜덤 시드

    Returns
    -------
    x_noisy : ndarray
        노이즈가 추가된 신호
    """
    if seed is not None:
        np.random.seed(seed)
    if snr_db is not None:
        sig_power = np.mean(x**2)
        noise_power = sig_power / (10**(snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(x))
    elif noise_std is not None:
        noise = noise_std * np.random.randn(len(x))
    else:
        raise ValueError("snr_db 또는 noise_std 중 하나를 지정하세요.")
    return x + noise


def bearing_fault_signal(fs, duration, f_fault, f_resonance,
                         damping=500, noise_level=0.5,
                         amp_variation=0.3, seed=42):
    """베어링 결함 시뮬레이션 신호 생성.

    Parameters
    ----------
    fs : float
        샘플링 주파수 (Hz)
    duration : float
        신호 길이 (초)
    f_fault : float
        결함 주파수 (Hz)
    f_resonance : float
        공진 주파수 (Hz)
    damping : float
        감쇠 계수 (기본 500)
    noise_level : float
        노이즈 표준편차 (기본 0.5)
    amp_variation : float
        충격 진폭 변동 (기본 0.3)
    seed : int
        랜덤 시드 (기본 42)

    Returns
    -------
    t : ndarray
        시간 축
    x_clean : ndarray
        순수 결함 신호
    x_noisy : ndarray
        노이즈 포함 결함 신호
    """
    np.random.seed(seed)
    t = np.arange(0, duration, 1/fs)
    N = len(t)
    n_impulses = int(duration * f_fault)

    x_clean = np.zeros(N)
    for k in range(n_impulses):
        t_impact = k / f_fault
        idx = int(t_impact * fs)
        if idx < N:
            amp = 1.0 + amp_variation * np.random.randn()
            decay = np.exp(-damping * np.maximum(t[idx:] - t_impact, 0))
            impulse = amp * decay * np.sin(
                2 * np.pi * f_resonance * (t[idx:] - t_impact))
            x_clean[idx:] += impulse[:N - idx]

    noise = noise_level * np.random.randn(N)
    x_noisy = x_clean + noise
    return t, x_clean, x_noisy

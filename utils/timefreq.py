"""
신호처리 실습 유틸리티 — 시간-주파수 분석 (STFT, 웨이블릿)
"""

import numpy as np


def stft(x, fs, nperseg=256, noverlap=None, nfft=None, window='hann'):
    """STFT 수행 — scipy.signal.spectrogram 래퍼.

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    nperseg : int
        윈도우 길이 (기본 256)
    noverlap : int, optional
        겹침 샘플 수 (기본: nperseg // 2)
    nfft : int, optional
        FFT 크기 (기본: nperseg)
    window : str
        윈도우 종류 (기본 'hann')

    Returns
    -------
    f : ndarray
        주파수 축
    t : ndarray
        시간 축
    Sxx : ndarray
        스펙트로그램 (파워)
    """
    from scipy.signal import spectrogram
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            nfft=nfft, window=window)
    return f, t, Sxx


def plot_spectrogram(f, t, Sxx, ax=None, db=True, log_freq=False,
                     cmap='viridis', vmin=None, vmax=None, **kwargs):
    """스펙트로그램 시각화.

    Parameters
    ----------
    f : ndarray
        주파수 축
    t : ndarray
        시간 축
    Sxx : ndarray
        스펙트로그램
    ax : matplotlib.axes.Axes, optional
        플롯 축
    db : bool
        dB 스케일 사용 여부 (기본 True)
    log_freq : bool
        로그 주파수축 사용 여부 (기본 False)
    cmap : str
        컬러맵 (기본 'viridis')
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    data = Sxx.copy()
    if db:
        data = 10 * np.log10(np.maximum(data, 1e-12))

    pcm = ax.pcolormesh(t, f, data, shading='gouraud', cmap=cmap,
                        vmin=vmin, vmax=vmax, **kwargs)

    if log_freq:
        ax.set_yscale('log')

    ax.set_xlabel('시간 (s)')
    ax.set_ylabel('주파수 (Hz)')

    return pcm


def cwt_morlet(x, fs, freq_range=(1, 500), n_scales=128):
    """Morlet CWT 수행 — pywt.cwt 래퍼.

    Parameters
    ----------
    x : ndarray
        입력 신호
    fs : float
        샘플링 주파수 (Hz)
    freq_range : tuple
        주파수 범위 (f_min, f_max) Hz
    n_scales : int
        스케일 수 (기본 128)

    Returns
    -------
    coeffs : ndarray
        CWT 계수 행렬 (n_scales, N)
    freqs : ndarray
        주파수 축 (Hz)
    """
    import pywt

    f_min, f_max = freq_range
    # Morlet 웨이블릿 중심 주파수
    central_freq = pywt.central_frequency('morl')

    # 주파수 범위에 대응하는 스케일 계산
    scales = central_freq * fs / np.linspace(f_max, f_min, n_scales)

    coeffs, freqs_out = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)
    return coeffs, freqs_out


def dwt_decompose(x, wavelet='db4', level=4):
    """DWT 다중해상도 분해.

    Parameters
    ----------
    x : ndarray
        입력 신호
    wavelet : str
        웨이블릿 종류 (기본 'db4')
    level : int
        분해 레벨 (기본 4)

    Returns
    -------
    coeffs : list
        [cA_n, cD_n, ..., cD_1] 계수 리스트
    """
    import pywt
    coeffs = pywt.wavedec(x, wavelet, level=level)
    return coeffs


def dwt_denoise(x, wavelet='db4', level=4, threshold='universal'):
    """DWT 기반 디노이징 (소프트 임계처리).

    Parameters
    ----------
    x : ndarray
        입력 신호
    wavelet : str
        웨이블릿 종류 (기본 'db4')
    level : int
        분해 레벨 (기본 4)
    threshold : str or float
        'universal' (기본) 또는 직접 임계값

    Returns
    -------
    x_denoised : ndarray
        디노이즈된 신호
    """
    import pywt

    coeffs = pywt.wavedec(x, wavelet, level=level)

    # 임계값 계산
    if threshold == 'universal':
        # 범용 임계값: sigma * sqrt(2 * ln(N))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = sigma * np.sqrt(2 * np.log(len(x)))
    else:
        thresh = threshold

    # 세부 계수에 소프트 임계처리 (근사 계수는 유지)
    denoised_coeffs = [coeffs[0]]  # 근사 계수 유지
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, thresh, mode='soft'))

    x_denoised = pywt.waverec(denoised_coeffs, wavelet)

    # 길이 맞추기
    if len(x_denoised) > len(x):
        x_denoised = x_denoised[:len(x)]

    return x_denoised

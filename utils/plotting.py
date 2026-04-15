"""
신호처리 실습 유틸리티 — 시각화 헬퍼
"""

import numpy as np
import matplotlib.pyplot as plt
from .style import COLORS, LINE_COLORS


def stem_plot(ax, x, y, color=None, markersize=4, linewidth=0.8, **kwargs):
    """stem plot 헬퍼 — 일관된 스타일.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        플롯 축
    x : array_like
        x축 데이터
    y : array_like
        y축 데이터
    color : str, optional
        색상 (기본: LINE_COLORS[0])
    markersize : int
        마커 크기 (기본 4)
    linewidth : float
        줄기 두께 (기본 0.8)
    """
    if color is None:
        color = LINE_COLORS[0]
    markerline, stemlines, baseline = ax.stem(x, y, **kwargs)
    plt.setp(stemlines, color=color, linewidth=linewidth)
    plt.setp(markerline, color=color, markersize=markersize)
    plt.setp(baseline, visible=False)


def plot_time_signal(t, x, ax=None, title='', xlabel='Time (s)', ylabel='Amplitude',
                     color=None, **kwargs):
    """시간 신호 플롯.

    Parameters
    ----------
    t : ndarray
        시간 축
    x : ndarray
        신호
    ax : matplotlib.axes.Axes, optional
        플롯 축
    title : str
        제목
    color : str, optional
        라인 색상
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if color is None:
        color = LINE_COLORS[0]
    ax.plot(t, x, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_spectrum(freq, amp, ax=None, title='', db=False, one_sided=True,
                  color=None, **kwargs):
    """주파수 스펙트럼 플롯.

    Parameters
    ----------
    freq : ndarray
        주파수 축
    amp : ndarray
        진폭 (또는 dB 값)
    ax : matplotlib.axes.Axes, optional
        플롯 축
    title : str
        제목
    db : bool
        True면 y축을 'dB'로 표시
    one_sided : bool
        True면 양의 주파수만 표시
    color : str, optional
        라인 색상
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    if color is None:
        color = LINE_COLORS[1]

    if one_sided:
        mask = freq >= 0
        freq = freq[mask]
        amp = amp[mask]

    ax.plot(freq, amp, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('dB' if db else 'Amplitude')
    return ax


def plot_time_and_spectrum(t, x, fs, figsize=(14, 5), title_time='Time Domain',
                           title_freq='Frequency Domain'):
    """시간 + 주파수 영역 나란히 플롯 — 가장 자주 쓰는 패턴.

    Parameters
    ----------
    t : ndarray
        시간 축
    x : ndarray
        신호
    fs : float
        샘플링 주파수

    Returns
    -------
    fig : Figure
    axes : tuple of (ax_time, ax_freq)
    """
    from .spectral import fft as _fft

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 시간 영역
    axes[0].plot(t, x, color=LINE_COLORS[0])
    axes[0].set_title(title_time)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    # 주파수 영역
    amp, freq = _fft(x, fs)
    axes[1].plot(freq, amp, color=LINE_COLORS[1])
    axes[1].set_title(title_freq)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')

    return fig, axes


def plot_fourier_coeffs(cn_dict, f0=None, title=''):
    """복소 푸리에 계수의 진폭/위상 스펙트럼.

    Parameters
    ----------
    cn_dict : dict
        {n: c_n} — 정수 키, 복소수(또는 SymPy) 값
    f0 : float or None
        None이면 x축 = 고조파 번호 n,
        숫자이면 x축 = n*f0 (Hz)
    title : str
        전체 제목
    """
    ns = np.array(sorted(cn_dict.keys()))
    mags = np.array([float(abs(cn_dict[n])) for n in ns])
    phases = np.array([float(np.angle(complex(cn_dict[n]))) for n in ns])
    phases[mags < 1e-10] = 0

    if f0 is not None:
        xs = ns * f0
        xlabel = 'Frequency (Hz)'
    else:
        xs = ns
        xlabel = '$n$'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 진폭
    ml, sl, bl = axes[0].stem(xs, mags)
    plt.setp(sl, color=LINE_COLORS[0])
    plt.setp(ml, color=LINE_COLORS[0])
    plt.setp(bl, color='gray')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('$|c_n|$')
    axes[0].set_title('Amplitude Spectrum')

    # 위상
    ml2, sl2, bl2 = axes[1].stem(xs, phases / np.pi)
    plt.setp(sl2, color=LINE_COLORS[1])
    plt.setp(ml2, color=LINE_COLORS[1])
    plt.setp(bl2, color='gray')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(r'$\angle c_n\, /\, \pi$')
    axes[1].set_title('Phase Spectrum')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_partial_sum(t, x_target, partial_sums, title=''):
    """원래 파형 + 여러 부분합을 겹쳐 그림.

    Parameters
    ----------
    t : ndarray
        시간 축
    x_target : ndarray
        원래 파형 (점선으로 표시)
    partial_sums : list of (ndarray, str)
        [(근사 파형, 라벨), ...]
    title : str
        제목
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, x_target, '--', color='gray', alpha=0.4, label='Original')

    for i, (approx, label) in enumerate(partial_sums):
        ax.plot(t, approx, color=LINE_COLORS[i % len(LINE_COLORS)],
                linewidth=1.5, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_comparison(signals, labels, t=None, fs=None, figsize=(12, 5)):
    """여러 신호 비교 플롯 (시간영역 겹쳐 그리기).

    Parameters
    ----------
    signals : list of ndarray
        신호 리스트
    labels : list of str
        범례 라벨 리스트
    t : ndarray, optional
        공통 시간 축
    fs : float, optional
        샘플링 주파수 (t가 없을 때 사용)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (sig, label) in enumerate(zip(signals, labels)):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        if t is not None:
            ax.plot(t, sig, color=color, label=label)
        else:
            ax.plot(sig, color=color, label=label)

    ax.legend()
    ax.set_xlabel('Time (s)' if t is not None else 'Sample')
    ax.set_ylabel('Amplitude')
    return fig, ax

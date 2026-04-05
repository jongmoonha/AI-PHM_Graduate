"""
신호처리 실습 유틸리티 패키지

사용법:
    from utils import fft, sine, chirp, add_noise, lowpass, bandpass
    from utils import hilbert_envelope, envelope_spectrum, cepstrum
    from utils import stft, cwt_morlet, dwt_decompose
    from utils import plot_time_and_spectrum, stem_plot
    from utils.style import setup_notebook, COLORS, LINE_COLORS
"""

from .spectral import fft, fft_full, ifft_real, to_db, psd_welch
from .signals import (sine, cosine, chirp, square_wave_fourier, add_noise,
                      composite_signal, bearing_fault_signal)
from .filters import lowpass, bandpass, fft_filter, moving_average, med_filter
from .timefreq import stft, plot_spectrogram, cwt_morlet, dwt_decompose, dwt_denoise
from .envelope import hilbert_envelope, envelope_spectrum, cepstrum
from .plotting import (stem_plot, plot_time_signal, plot_spectrum,
                       plot_time_and_spectrum, plot_comparison,
                       plot_fourier_coeffs, plot_partial_sum)
from .style import setup_notebook, COLORS, LINE_COLORS

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import sys
import subprocess

COLORS = {
    'primary': '#1B4F72',
    'warning': '#C0392B',
    'code_bg': '#F2F3F4',
    'concept_bg': '#EBF5FB',
    'table_header': '#2C3E50',
    'line1': '#2980B9',
    'line2': '#E74C3C',
    'line3': '#27AE60',
    'line4': '#F39C12',
    'line5': '#8E44AD',
}

LINE_COLORS = [COLORS['line1'], COLORS['line2'], COLORS['line3'],
               COLORS['line4'], COLORS['line5']]


def setup_notebook():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.prop_cycle': plt.cycler(color=LINE_COLORS),
        'legend.loc': 'upper right',
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'figure.constrained_layout.use': True,
    })
    _setup_korean_font()
    plt.rcParams['axes.unicode_minus'] = False


def _setup_korean_font():
    candidates = []
    if os.name == 'nt':
        candidates = ['Malgun Gothic', 'NanumGothic']
    elif sys.platform == 'darwin':
        candidates = ['AppleGothic', 'NanumGothic']
    else:
        candidates = ['NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans KR']

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams['font.family'] = name
            return

    # Colab/Linux fallback: install Nanum and register
    if 'google.colab' in sys.modules or sys.platform.startswith('linux'):
        try:
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'fonts-nanum'],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            font_paths = fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum'])
            for fp in font_paths:
                fm.fontManager.addfont(fp)
            if any('Nanum' in f.name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = 'NanumGothic'
        except Exception:
            pass

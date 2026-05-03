# AI-PHM_Graduate — 신호처리 실습

AI 기반 PHM(Prognostics and Health Management) 대학원 강의의 신호처리 실습 노트북 모음.
Python 기초부터 푸리에 해석, 필터링, 시간-주파수 분석, 진동 신호 기반 결함 진단까지 다룹니다.

## 사용법

각 노트북 첫 셀에 **Colab 부트스트랩** 이 포함되어 있어,
Colab 에서는 자동으로 본 repo 를 clone 하고 작업 디렉토리를 맞춥니다.
별도 설정 없이 `import utils`, `pd.read_csv('data/...')` 가 그대로 동작합니다.

### 로컬 실행
```bash
git clone https://github.com/jongmoonha/AI-PHM_Graduate.git
cd AI-PHM_Graduate
pip install -r requirements.txt
jupyter lab
```

### Colab 실행
아래 "실습 목록" 표의 Colab 뱃지를 클릭하면 바로 실행됩니다.

## 실습 목록

| 노트북 | Colab |
|--------|-------|
| [[a_Practice_Python 1] Basics.ipynb](%5Ba_Practice_Python%201%5D%20Basics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Ba_Practice_Python%201%5D%20Basics.ipynb) |
| [[a_Practice_Python 2] Data Load and Plot.ipynb](%5Ba_Practice_Python%202%5D%20Data%20Load%20and%20Plot.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Ba_Practice_Python%202%5D%20Data%20Load%20and%20Plot.ipynb) |
| [[c_advanced] a_Fourier_Series.ipynb](%5Bc_advanced%5D%20a_Fourier_Series.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20a_Fourier_Series.ipynb) |
| [[c_advanced] b_Sampling_DFT.ipynb](%5Bc_advanced%5D%20b_Sampling_DFT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20b_Sampling_DFT.ipynb) |
| [[b_basics] a_FFT.ipynb](%5Bb_basics%5D%20a_FFT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bb_basics%5D%20a_FFT.ipynb) |
| [[b_basics] b_Filtering.ipynb](%5Bb_basics%5D%20b_Filtering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bb_basics%5D%20b_Filtering.ipynb) |
| [[c_advanced] c_Spectrum_Analysis.ipynb](%5Bc_advanced%5D%20c_Spectrum_Analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20c_Spectrum_Analysis.ipynb) |
| [[b_basics] c_Time_frequency.ipynb](%5Bb_basics%5D%20c_Time_frequency.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bb_basics%5D%20c_Time_frequency.ipynb) |
| [[c_advanced] d_Time_Frequency.ipynb](%5Bc_advanced%5D%20d_Time_Frequency.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20d_Time_Frequency.ipynb) |
| [[b_basics] d_Envelope.ipynb](%5Bb_basics%5D%20d_Envelope.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bb_basics%5D%20d_Envelope.ipynb) |
| [[c_advanced] e_Deconvolution.ipynb](%5Bc_advanced%5D%20e_Deconvolution.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20e_Deconvolution.ipynb) |
| [[c_advanced] f_Resampling_OrderAnalysis_TSA.ipynb](%5Bc_advanced%5D%20f_Resampling_OrderAnalysis_TSA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bc_advanced%5D%20f_Resampling_OrderAnalysis_TSA.ipynb) |
| [[b_basics] e_Feature_CWRU_DE_IR.ipynb](%5Bb_basics%5D%20e_Feature_CWRU_DE_IR.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jongmoonha/AI-PHM_Graduate/blob/main/%5Bb_basics%5D%20e_Feature_CWRU_DE_IR.ipynb) |

## 폴더 구조

```
AI-PHM_Graduate/
├── [a_Practice_Python 1] Basics.ipynb
├── [a_Practice_Python 2] Data Load and Plot.ipynb
├── [b_basics] a_FFT.ipynb
├── ...
├── [c_advanced] f_Resampling_OrderAnalysis_TSA.ipynb
├── utils.py            # 실습 공통 헬퍼 (FFT, 필터링, 포락선, 특징량 등)
├── kurtogram.py        # Fast Kurtogram 구현 (외부 코드)
├── data/               # 실습 데이터 (CSV, MP3)
└── requirements.txt
```

## 의존성

`requirements.txt` 참조. Colab 기본 환경에서 대부분 동작하며, 추가로 필요한 패키지만 명시했습니다.

## 라이선스
교육용. 외부 코드 `kurtogram.py` 의 원저작권은 해당 저자에게 있습니다.

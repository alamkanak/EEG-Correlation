# TMS-EEG-MEP-Correlation

This project calculates PSD and phases of EEG datasets. The input files are what outputted by https://github.com/alamkanak/EEG-Processing-Matlab and the output files are excel files containing powers and phases. Please read the `README.md` of https://github.com/alamkanak/EEG-Processing-Matlab before starting to read this document.

## Install
1. Create a conda environment with python 3.7
2. Run the commands mentioned in [requirements.txt](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/requirements.txt) file
3. Download the datasets in `data` folder
4. Open jupyter notebook

## Dataset 1
The file [`164-d1.ipynb`](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/164-d1.ipynb) does all the processing of dataset1. It reads four types of files and processes them:

- Cleaned Hjorth transformed files: `06-clean-prestimulus-hjorth.mat`
- Cleaned Raw EEG files: `06-clean-prestimulus.p`
- Artifactual Hjorth transformed files: `010-raw-hjorth.mat`
- Artifactual raw EEG files: `raw.p`

The processing outputs are stored in `164-d1-powers.csv` and `164-d1-phases.csv` files. The output files are not stored in the repository for large filesize. The output files are further processed in Rstudio.

## Dataset 2
The file [`157-d2.ipynb`](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/157-d2.ipynb) does all the processing of dataset2. It reads 2 types of files and processes them:

- Cleaned Hjorth transformed files: `*-hjorth.mat`
- Cleaned Raw EEG files: `*.csv`

The processing outputs are stored in `157-alc-power-long.xlsx` and `157-alc-phase-long.xlsx` files. The output files are not stored in the repository for large filesize. The output files are further processed in Rstudio.

## Dataset 3
The file [`166-d3.ipynb`](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/166-d3.ipynb) does all the processing of dataset3. It reads four types of files and processes them:

- Cleaned Hjorth transformed files: `clean-hjorth/*.mat`
- Cleaned Raw EEG files: `clean/*.mat`
- Artifactual Hjorth transformed files: `raw-hjorth/*.mat`
- Artifactual raw EEG files: `raw/*.mat`

The processing outputs are stored in `157-alc-power-long.xlsx` and `157-alc-phase-long.xlsx` files. The output files are not stored in the repository for large filesize. The output files are further processed in Rstudio.

## Statistical analysis
The correlation between methodological choices and power-phase quantities were performed separately in different files for different datasets.
- Dataset 1: [d1.R](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/StatisticalAnalysis/d1.R)
- Dataset 2: [d2.R](https://github.com/alamkanak/TMS-EEG-MEP-Correlation/blob/master/StatisticalAnalysis/d2.R)
- Dataset 3: [d3.R]

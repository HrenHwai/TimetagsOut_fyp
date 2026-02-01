# TimetagsOut.py - TTM8000 Time-Tag Data Prcessing Pipeline

this repository contains a Python script ('TimetagsOut.py') used to 
post-process high precision timetag data acquired from TTM8000
time-tagging module. It was developed for the path - selection QRNG 
experiment.

## What this script does 
### 1) Load and inspect time-tag data
- read HDF5 datasets such as, Channel, ProcessedTimetag and other
  timetag-related field if present

- provides quick previews to verify the input fomat and content

## 2) Generate raw bitstreams from two detector channels
- converts detector events into bits by time-window binning
- keeps only windows with exactly one detecyion
- bit assignment:
  ch_1 -> 1
  ch_2 -> 0

## 3) post-processing / randomness extraction options
- Von-Neumann correction
- SHA-256 hashing per block
- Toeplitz hashing (extractor compression using an estimated min-entropy)

## 4) Quality metrics and analysis
- Bias per block: \|P(1) - 0.5\|
- min-entropy estimate
- temporal correlation analysis (autocorrelation or fft-based correlation)

## Dependencies 
install common scientifuc python packages:
'''bash
pip install numpy pandas h5py scipy matplotlib

## Project structure
- `src/timetagsout/` — core Python implementation (`TimetagsOut`)
- `examples/` — Jupyter notebook(s) demonstrating usage and analysis

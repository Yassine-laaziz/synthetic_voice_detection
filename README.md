# AI Voice Detector

Research on detecting AI-generated voices. Focus on deepfake detection and understanding what makes synthetic speech different from human speech.

## What this does

- Audio feature extraction (MFCC, spectral, temporal, harmonic)
- Statistical analysis and comparison
- Machine learning models for detection (Random Forest, SVM)

## Research areas

1. Speech physics and biology - how human speech actually works
2. TTS internals - how neural TTS systems work under the hood
3. Hardware evaluation - do we need better microphones for detection?

## Structure

```
src/
├── audio_processing/    # feature extraction
├── analysis/            # statistical analysis
└── models/              # detection models
tests/                   # unit tests
data/                    # scraped papers, databases
research/                # research notes and framework
```

## Setup

```bash
pip install -r requirements.txt
```

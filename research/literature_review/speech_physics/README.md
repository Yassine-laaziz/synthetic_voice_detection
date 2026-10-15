# Speech Physics and Biology

## Goal

Understand how human speech actually works physically. Need this to figure out what TTS systems can't or don't capture.

## What to study

- Source-filter theory (glottal source + vocal tract filter)
- Vocal cord physics and biomechanics
- Frequency ranges: sub-50Hz, fundamental, formants, ultrasonic
- Temporal dynamics: micro-timing, jitter, shimmer

## Why this matters

If we know what natural speech looks like physically, we can identify what TTS violates. This is needed before answering the hardware question.

## Search terms

- source-filter theory speech production
- vocal tract acoustics
- vocal cord vibration biomechanics
- formant frequencies speech
- speech production biomechanics
- sub-50Hz speech analysis
- ultrasonic speech
- micro-timing speech
- jitter shimmer voice
- prosodic timing

## Scraping

```bash
cd data
python scrape_speech_physics.py
```

Saves to `data/speech_physics_papers.json`

# TTS Internals and Architecture

## Goal

Understand how neural TTS systems work internally. Not just the output, but the architecture, training, and what they can't do.

## What to study

- Neural architectures (autoregressive vs parallel, attention, receptive fields)
- Training objectives (likelihood, adversarial, reconstruction losses)
- Latent space structure (VAE, flow-based, manifold)
- Information-theoretic properties (entropy, mutual information)
- Optimization artifacts (gradient descent, convergence)

## Why this matters

If we understand the internals, we can find fundamental violations of natural speech, not just surface artifacts.

## Key TTS systems

- WaveNet (2016) - autoregressive
- WaveGlow (2019) - flow-based
- Tacotron (2017) - encoder-decoder with attention
- VITS (2021) - variational + flows
- YourTTS (2021) - zero-shot cloning
- NaturalSpeech (2022) - human-level quality

## Search terms

- neural TTS architecture limitations
- attention mechanism text-to-speech
- neural vocoder architecture
- TTS training objectives
- latent space text-to-speech
- variational autoencoder speech synthesis
- flow-based speech synthesis
- entropy speech synthesis
- information theory TTS
- optimization artifacts neural TTS

## Scraping

```bash
cd data
python scrape_tts_internals.py
```

Saves to `data/tts_internals_papers.json`

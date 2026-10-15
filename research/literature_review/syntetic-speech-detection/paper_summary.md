# Paper Summaries Summary

The `all_paper_summaries.txt` file contains extracted text from 10 research papers on AI voice detection. Here's a summary:

## Papers

1. ASVspoof 2021 - Challenge overview with three tasks: Logical Access (synthetic speech), Physical Access (replay attacks), and Deepfake (voice cloning). Results show progress but physical access remains difficult.

2. ASVspoof 2019 - Earlier challenge focusing on logical access with state-of-the-art neural TTS systems.

3. SpoTNet (2023) - Transformer-based detection architecture achieving EER 0.95% on ASVspoof-LA.

4. End-to-End Detection (2021) - CQT/CQCC features for synthetic speech detection.

5. Spectro-Temporal Graph Attention Networks - Graph-based detection modeling spectro-temporal relationships.

6. MFCC Detection - Baseline MFCC-based approach using machine learning.

7. Neural Vocoder Artifacts (2023) - Detecting synthesis artifacts from neural vocoders.

8. Advances in Anti-Spoofing - Survey paper covering the field's evolution.

9. Additional TTS papers - Papers on newer TTS techniques and detection methods.

10. ASVspoof 5 (2024) - Latest challenge with crowdsourced data.

## Key Themes

- ASVspoof is the primary benchmark across all papers
- Detection methods evolved from simple features (MFCC) to deep learning (Transformers, Graph networks)
- Generalization to unseen attacks remains the major challenge
- Physical access (replay) is harder to detect than logical access (synthetic)
- Deepfake detection is a newer, evolving area

## Gaps Identified

- No foundational TTS generation papers (WaveNet, Tacotron, VITS)
- No hardware/recording quality papers
- Generalization problem unsolved

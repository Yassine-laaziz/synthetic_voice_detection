"""
Feature extraction for audio analysis
"""

import librosa
import numpy as np
from typing import Dict, Tuple, Optional


class FeatureExtractor:
    """Extract acoustic features from audio files for AI voice detection"""

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Initialize feature extractor

        Args:
            sample_rate: Target sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr

    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            MFCC coefficients (n_mfcc, n_frames)
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        return mfcc

    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract spectral features

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dictionary of spectral features
        """
        features = {
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr)[0],
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr)[0],
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0],
            'spectral_contrast': librosa.feature.spectral_contrast(y=audio, sr=sr)[0],
            'spectral_flatness': librosa.feature.spectral_flatness(y=audio)[0],
        }
        return features

    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract temporal features

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dictionary of temporal features
        """
        features = {
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)[0],
            'rms_energy': librosa.feature.rms(y=audio)[0],
        }

        # Pitch (F0) extraction
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch)
        features['pitch'] = np.array(pitch_values)

        return features

    def extract_harmonic_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract harmonic features

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dictionary of harmonic features
        """
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(audio)

        features = {
            'harmonic_component': y_harmonic,
            'percussive_component': y_percussive,
        }

        return features

    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing all extracted features
        """
        audio, sr = self.load_audio(audio_path)

        features = {
            'mfcc': self.extract_mfcc(audio, sr),
            'spectral': self.extract_spectral_features(audio, sr),
            'temporal': self.extract_temporal_features(audio, sr),
            'harmonic': self.extract_harmonic_features(audio, sr),
        }

        return features

    def extract_feature_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics for extracted features

        Args:
            features: Dictionary of features

        Returns:
            Dictionary of feature statistics (mean, std, min, max)
        """
        stats = {}

        # MFCC statistics
        mfcc = features['mfcc']
        for i in range(mfcc.shape[0]):
            stats[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            stats[f'mfcc_{i}_std'] = np.std(mfcc[i])

        # Spectral feature statistics
        for name, feat in features['spectral'].items():
            if feat.ndim == 1:
                stats[f'{name}_mean'] = np.mean(feat)
                stats[f'{name}_std'] = np.std(feat)
            else:
                stats[f'{name}_mean'] = np.mean(feat)
                stats[f'{name}_std'] = np.std(feat)

        # Temporal feature statistics
        for name, feat in features['temporal'].items():
            if name == 'pitch':
                # Filter out unvoiced frames (pitch = 0)
                valid_pitches = feat[feat > 0]
                if len(valid_pitches) > 0:
                    stats[f'{name}_mean'] = np.mean(valid_pitches)
                    stats[f'{name}_std'] = np.std(valid_pitches)
                    stats[f'{name}_range'] = np.max(valid_pitches) - np.min(valid_pitches)
            else:
                stats[f'{name}_mean'] = np.mean(feat)
                stats[f'{name}_std'] = np.std(feat)

        return stats

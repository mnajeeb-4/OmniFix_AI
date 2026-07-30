"""
Audio processing module for spectral analysis and fault detection.
Uses librosa for MFCC extraction, pitch detection, and frequency deviation analysis.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """Load audio file and return signal and sample rate."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def extract_mfcc(self, y, sr, n_mfcc=13):
        """Extract MFCC features from audio signal."""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs

    def detect_pitch_deviation(self, y, sr):
        """Detect pitch/frequency deviation using spectral centroid and F0."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Get the dominant pitch for each frame
        pitch = pitches[magnitudes.argmax(axis=0)]
        # Remove zeros (silence)
        pitch = pitch[pitch > 0]
        if len(pitch) == 0:
            return 0.0
        mean_pitch = np.mean(pitch)
        std_pitch = np.std(pitch)
        deviation = (std_pitch / mean_pitch) * 100  # percentage
        return deviation

    def analyze_bearing(self, y, sr):
        """
        Detect bearing faults by analyzing high‑frequency noise.
        Bearing defects often produce harmonics above 2 kHz.
        """
        # Compute spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_cent = np.mean(cent)
        # Compute high‑frequency energy ratio (>2000 Hz)
        fft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_mask = freqs > 2000
        high_energy = np.sum(fft[high_freq_mask, :])
        total_energy = np.sum(fft)
        high_ratio = high_energy / total_energy if total_energy > 0 else 0
        # Heuristic: if spectral centroid is high and high‑frequency ratio is elevated -> bearing issue
        if mean_cent > 2000 and high_ratio > 0.3:
            return {"fault": "Bearing wear", "severity": "WARNING", "confidence": 0.75}
        elif mean_cent > 1500 and high_ratio > 0.2:
            return {"fault": "Possible bearing wear", "severity": "MODERATE", "confidence": 0.55}
        else:
            return {"fault": "Healthy", "severity": "HEALTHY", "confidence": 0.9}

    def analyze_misfire(self, y, sr):
        """
        Detect engine misfire by looking for irregular low‑frequency pulsing.
        Misfires cause sudden energy drops at the fundamental firing frequency.
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Detect peaks (irregularities)
        onset_peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        # If too many or too few peaks per second, it might indicate misfire
        duration = len(y) / sr
        peaks_per_sec = len(onset_peaks) / duration
        # Normal engine idle ~ 10-20 peaks per sec (approx 600-1200 RPM)
        # Misfire often causes missing peaks or erratic pattern
        if peaks_per_sec < 5 or peaks_per_sec > 40:
            return {"fault": "Irregular firing pattern - possible misfire", "severity": "CRITICAL", "confidence": 0.82}
        else:
            return {"fault": "Normal firing pattern", "severity": "HEALTHY", "confidence": 0.85}

    def analyze_belt_squeak(self, y, sr):
        """
        Detect belt squeak by looking for high‑frequency tonal components above 2 kHz.
        """
        # Compute spectral flatness (tonality measure)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flatness = np.mean(flatness)
        # High‑frequency energy ratio
        fft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_mask = (freqs > 2000) & (freqs < 5000)
        high_energy = np.sum(fft[high_freq_mask, :])
        total_energy = np.sum(fft)
        high_ratio = high_energy / total_energy if total_energy > 0 else 0
        # Belt squeak often has low flatness (tonal) and high high‑freq energy
        if mean_flatness < 0.1 and high_ratio > 0.4:
            return {"fault": "Belt squeak detected", "severity": "WARNING", "confidence": 0.78}
        elif mean_flatness < 0.2 and high_ratio > 0.25:
            return {"fault": "Possible belt issue", "severity": "MODERATE", "confidence": 0.55}
        else:
            return {"fault": "Healthy", "severity": "HEALTHY", "confidence": 0.9}

    def generate_spectrogram(self, y, sr):
        """Generate a spectrogram plot and return as base64 image."""
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram (dB)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

    def full_analysis(self, audio_file):
        """
        Run all analyses and return a combined report.
        """
        y, sr = self.load_audio(audio_file)
        bearing_result = self.analyze_bearing(y, sr)
        misfire_result = self.analyze_misfire(y, sr)
        belt_result = self.analyze_belt_squeak(y, sr)
        spectrogram = self.generate_spectrogram(y, sr)
        pitch_dev = self.detect_pitch_deviation(y, sr)
        return {
            "bearing": bearing_result,
            "misfire": misfire_result,
            "belt": belt_result,
            "spectrogram": spectrogram,
            "pitch_deviation_pct": pitch_dev
        }

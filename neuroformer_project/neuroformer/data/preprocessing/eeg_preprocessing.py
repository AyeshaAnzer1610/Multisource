"""
EEG Preprocessing Pipeline

Implements the full preprocessing workflow described in the paper:
1. Bandpass filtering (0.5-50 Hz)
2. ICA-based artifact removal
3. Epoch segmentation
4. Feature extraction (time-domain, frequency-domain, connectivity)
"""

import numpy as np
import mne
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """EEG preprocessing pipeline."""
    
    def __init__(
        self,
        sampling_rate: int = 500,
        n_channels: int = 5,
        channel_names: List[str] = ['Fp1', 'Fp2', 'C3', 'C4', 'O1'],
        lowcut: float = 0.5,
        highcut: float = 50.0,
        epoch_length: float = 1.0,  # seconds
        n_ica_components: int = 5
    ):
        """
        Initialize EEG preprocessor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            n_channels: Number of EEG channels
            channel_names: Names of EEG channels
            lowcut: Lower frequency for bandpass filter
            highcut: Upper frequency for bandpass filter
            epoch_length: Length of each epoch in seconds
            n_ica_components: Number of ICA components
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.channel_names = channel_names
        self.lowcut = lowcut
        self.highcut = highcut
        self.epoch_length = epoch_length
        self.epoch_samples = int(epoch_length * sampling_rate)
        self.n_ica_components = n_ica_components
        
        # Frequency bands for spectral features
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply 4th-order Butterworth bandpass filter.
        
        Args:
            data: Raw EEG data (n_samples, n_channels)
        
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=0)
        
        return filtered
    
    def apply_ica(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ICA for artifact removal.
        
        Args:
            data: Filtered EEG data (n_samples, n_channels)
        
        Returns:
            Cleaned data and ICA components
        """
        # Standardize data
        data_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)
        
        # Apply ICA
        ica = FastICA(n_components=self.n_ica_components, random_state=42, max_iter=500)
        components = ica.fit_transform(data_std)
        
        # Identify artifact components (simplified heuristic)
        # In practice, this would use more sophisticated criteria
        artifact_components = self._identify_artifacts(components)
        
        # Remove artifact components
        components[:, artifact_components] = 0
        
        # Reconstruct signal
        cleaned = ica.inverse_transform(components)
        
        # Rescale
        cleaned = cleaned * data.std(axis=0) + data.mean(axis=0)
        
        return cleaned, components
    
    def _identify_artifacts(self, components: np.ndarray) -> List[int]:
        """
        Identify artifact components using statistical criteria.
        
        Args:
            components: ICA components (n_samples, n_components)
        
        Returns:
            List of artifact component indices
        """
        artifacts = []
        
        for i in range(components.shape[1]):
            comp = components[:, i]
            
            # Check kurtosis (high kurtosis suggests artifacts)
            kurt = kurtosis(comp)
            if kurt > 5:
                artifacts.append(i)
                continue
            
            # Check frequency content (high-frequency artifacts)
            freqs, psd = signal.welch(comp, self.sampling_rate, nperseg=256)
            high_freq_power = psd[freqs > 20].sum()
            total_power = psd.sum()
            if high_freq_power / total_power > 0.5:
                artifacts.append(i)
        
        return artifacts
    
    def segment_epochs(self, data: np.ndarray) -> np.ndarray:
        """
        Segment continuous data into fixed-length epochs.
        
        Args:
            data: Continuous EEG data (n_samples, n_channels)
        
        Returns:
            Epoched data (n_epochs, epoch_samples, n_channels)
        """
        n_samples = data.shape[0]
        n_epochs = n_samples // self.epoch_samples
        
        # Truncate to fit whole epochs
        data_truncated = data[:n_epochs * self.epoch_samples]
        
        # Reshape into epochs
        epochs = data_truncated.reshape(n_epochs, self.epoch_samples, self.n_channels)
        
        return epochs
    
    def extract_time_domain_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features from an epoch.
        
        Args:
            epoch: Single epoch (epoch_samples, n_channels)
        
        Returns:
            Time-domain features (25 dimensions: 5 features × 5 channels)
        """
        features = []
        
        for ch in range(self.n_channels):
            signal_ch = epoch[:, ch]
            
            features.extend([
                np.mean(signal_ch),           # Mean amplitude
                np.std(signal_ch),            # Standard deviation
                np.sqrt(np.mean(signal_ch**2)),  # RMS
                skew(signal_ch),              # Skewness
                kurtosis(signal_ch)           # Kurtosis
            ])
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract frequency-domain features from an epoch.
        
        Args:
            epoch: Single epoch (epoch_samples, n_channels)
        
        Returns:
            Frequency-domain features (100 dimensions: 20 features × 5 channels)
        """
        features = []
        
        for ch in range(self.n_channels):
            signal_ch = epoch[:, ch]
            
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(
                signal_ch,
                fs=self.sampling_rate,
                nperseg=self.epoch_samples,
                noverlap=self.epoch_samples // 2,
                window='hamming'
            )
            
            # Extract band powers
            band_powers = {}
            total_power = psd.sum()
            
            for band_name, (low, high) in self.freq_bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = psd[band_mask].sum()
                band_powers[band_name] = band_power
                
                # Absolute power
                features.append(band_power)
                
                # Relative power
                features.append(band_power / (total_power + 1e-10))
            
            # Band power ratios
            features.append(band_powers['theta'] / (band_powers['alpha'] + 1e-10))
            features.append(band_powers['beta'] / (band_powers['alpha'] + 1e-10))
            
            # Spectral entropy
            psd_norm = psd / (psd.sum() + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            features.append(spectral_entropy)
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            features.append(peak_freq)
        
        return np.array(features)
    
    def extract_connectivity_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract cross-channel connectivity features.
        
        Args:
            epoch: Single epoch (epoch_samples, n_channels)
        
        Returns:
            Connectivity features (3 dimensions)
        """
        features = []
        
        # Channel pairs for connectivity
        # Fp1-C3 (frontal-central)
        # C3-C4 (inter-hemispheric)
        # Fp1-O1 (frontal-occipital)
        
        pairs = [(0, 2), (2, 3), (0, 4)]  # Corresponding to channel indices
        
        for ch1, ch2 in pairs:
            signal1 = epoch[:, ch1]
            signal2 = epoch[:, ch2]
            
            # Compute coherence in alpha band
            freqs, coherence = signal.coherence(
                signal1, signal2,
                fs=self.sampling_rate,
                nperseg=self.epoch_samples
            )
            
            # Alpha band coherence
            alpha_mask = (freqs >= 8) & (freqs < 13)
            alpha_coherence = coherence[alpha_mask].mean()
            
            features.append(alpha_coherence)
        
        return np.array(features)
    
    def extract_features_from_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single epoch.
        
        Args:
            epoch: Single epoch (epoch_samples, n_channels)
        
        Returns:
            Feature vector (128 dimensions)
        """
        time_features = self.extract_time_domain_features(epoch)  # 25 dims
        freq_features = self.extract_frequency_domain_features(epoch)  # 100 dims
        conn_features = self.extract_connectivity_features(epoch)  # 3 dims
        
        # Concatenate all features
        features = np.concatenate([time_features, freq_features, conn_features])
        
        return features
    
    def preprocess(
        self,
        raw_data: np.ndarray,
        apply_ica: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline.
        
        Args:
            raw_data: Raw EEG data (n_samples, n_channels)
            apply_ica: Whether to apply ICA artifact removal
        
        Returns:
            Dictionary containing:
                - features: Extracted features (n_epochs, 128)
                - epochs: Segmented epochs (n_epochs, epoch_samples, n_channels)
                - cleaned: Artifact-removed data (n_samples, n_channels)
        """
        # 1. Bandpass filtering
        filtered = self.bandpass_filter(raw_data)
        
        # 2. ICA artifact removal
        if apply_ica:
            cleaned, components = self.apply_ica(filtered)
        else:
            cleaned = filtered
            components = None
        
        # 3. Segment into epochs
        epochs = self.segment_epochs(cleaned)
        
        # 4. Extract features from each epoch
        n_epochs = epochs.shape[0]
        features = np.zeros((n_epochs, 128))
        
        for i in range(n_epochs):
            features[i] = self.extract_features_from_epoch(epochs[i])
        
        return {
            'features': features,
            'epochs': epochs,
            'cleaned': cleaned,
            'components': components
        }


def preprocess_eeg_file(
    filepath: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Preprocess a single EEG file.
    
    Args:
        filepath: Path to EEG file (.edf format)
        output_path: Optional path to save features
        **kwargs: Additional arguments for EEGPreprocessor
    
    Returns:
        Preprocessing results dictionary
    """
    # Load EEG data using MNE
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    # Get data
    data = raw.get_data().T  # (n_samples, n_channels)
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(**kwargs)
    
    # Preprocess
    results = preprocessor.preprocess(data)
    
    # Save if requested
    if output_path:
        np.savez(
            output_path,
            features=results['features'],
            epochs=results['epochs']
        )
    
    return results


if __name__ == "__main__":
    # Test preprocessing
    # Generate synthetic EEG data
    sampling_rate = 500
    duration = 60  # seconds
    n_samples = sampling_rate * duration
    n_channels = 5
    
    # Simulate EEG with multiple frequency components
    t = np.arange(n_samples) / sampling_rate
    data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Delta (1 Hz)
        data[:, ch] += 2.0 * np.sin(2 * np.pi * 1.0 * t)
        # Theta (6 Hz)
        data[:, ch] += 1.5 * np.sin(2 * np.pi * 6.0 * t)
        # Alpha (10 Hz)
        data[:, ch] += 1.0 * np.sin(2 * np.pi * 10.0 * t)
        # Beta (20 Hz)
        data[:, ch] += 0.5 * np.sin(2 * np.pi * 20.0 * t)
        # Add noise
        data[:, ch] += 0.2 * np.random.randn(n_samples)
    
    # Preprocess
    preprocessor = EEGPreprocessor()
    results = preprocessor.preprocess(data)
    
    print(f"Features shape: {results['features'].shape}")
    print(f"Epochs shape: {results['epochs'].shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {results['features'].mean():.3f}")
    print(f"  Std: {results['features'].std():.3f}")
    print(f"  Min: {results['features'].min():.3f}")
    print(f"  Max: {results['features'].max():.3f}")

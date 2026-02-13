"""
Eye-Tracking Preprocessing Pipeline

Implements preprocessing and feature extraction for eye-tracking data:
1. Parse raw gaze coordinates and pupil diameter
2. Identify fixations, saccades, and blinks
3. Extract comprehensive eye-tracking features (96 dimensions)
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EyeTrackingPreprocessor:
    """Eye-tracking preprocessing and feature extraction."""
    
    def __init__(
        self,
        sampling_rate: int = 60,  # Hz
        window_length: float = 1.0,  # seconds
        fixation_threshold: float = 30.0,  # degrees/sec
        fixation_min_duration: float = 0.1,  # seconds
        saccade_threshold: float = 30.0,  # degrees/sec
        blink_max_duration: float = 0.5  # seconds
    ):
        """
        Initialize eye-tracking preprocessor.
        
        Args:
            sampling_rate: Eye-tracker sampling frequency
            window_length: Feature extraction window length
            fixation_threshold: Velocity threshold for fixation detection
            fixation_min_duration: Minimum fixation duration
            saccade_threshold: Velocity threshold for saccade detection
            blink_max_duration: Maximum blink duration
        """
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.window_samples = int(window_length * sampling_rate)
        self.fixation_threshold = fixation_threshold
        self.fixation_min_duration = fixation_min_duration
        self.saccade_threshold = saccade_threshold
        self.blink_max_duration = blink_max_duration
    
    def compute_velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gaze velocity from position.
        
        Args:
            x, y: Gaze coordinates in degrees
        
        Returns:
            Velocity in degrees/second
        """
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        
        velocity = np.sqrt(dx**2 + dy**2) * self.sampling_rate
        
        return velocity
    
    def detect_fixations(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Detect fixations using velocity threshold.
        
        Args:
            x, y: Gaze coordinates
        
        Returns:
            List of (start_idx, end_idx) tuples for each fixation
        """
        velocity = self.compute_velocity(x, y)
        
        # Fixation periods: velocity < threshold
        is_fixation = velocity < self.fixation_threshold
        
        # Find continuous fixation periods
        fixations = []
        start_idx = None
        
        for i, is_fix in enumerate(is_fixation):
            if is_fix and start_idx is None:
                start_idx = i
            elif not is_fix and start_idx is not None:
                # Check duration
                duration = (i - start_idx) / self.sampling_rate
                if duration >= self.fixation_min_duration:
                    fixations.append((start_idx, i))
                start_idx = None
        
        # Handle final fixation
        if start_idx is not None:
            duration = (len(is_fixation) - start_idx) / self.sampling_rate
            if duration >= self.fixation_min_duration:
                fixations.append((start_idx, len(is_fixation)))
        
        return fixations
    
    def detect_saccades(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fixations: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Detect saccades as high-velocity periods between fixations.
        
        Args:
            x, y: Gaze coordinates
            fixations: List of fixation periods
        
        Returns:
            List of (start_idx, end_idx) tuples for each saccade
        """
        velocity = self.compute_velocity(x, y)
        
        saccades = []
        
        # Saccades occur between fixations
        for i in range(len(fixations) - 1):
            fix_end = fixations[i][1]
            next_fix_start = fixations[i + 1][0]
            
            if next_fix_start > fix_end:
                # Check if velocity exceeds threshold
                segment_velocity = velocity[fix_end:next_fix_start]
                if np.max(segment_velocity) >= self.saccade_threshold:
                    saccades.append((fix_end, next_fix_start))
        
        return saccades
    
    def detect_blinks(self, pupil_left: np.ndarray, pupil_right: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect blinks as periods with missing pupil data.
        
        Args:
            pupil_left, pupil_right: Pupil diameter for each eye
        
        Returns:
            List of (start_idx, end_idx) tuples for each blink
        """
        # Missing data indicated by NaN or zero
        missing = np.isnan(pupil_left) | np.isnan(pupil_right) | \
                  (pupil_left == 0) | (pupil_right == 0)
        
        blinks = []
        start_idx = None
        
        for i, is_missing in enumerate(missing):
            if is_missing and start_idx is None:
                start_idx = i
            elif not is_missing and start_idx is not None:
                # Check duration
                duration = (i - start_idx) / self.sampling_rate
                if duration <= self.blink_max_duration:
                    blinks.append((start_idx, i))
                start_idx = None
        
        # Handle final blink
        if start_idx is not None:
            duration = (len(missing) - start_idx) / self.sampling_rate
            if duration <= self.blink_max_duration:
                blinks.append((start_idx, len(missing)))
        
        return blinks
    
    def extract_fixation_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fixations: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Extract fixation-related features (24 dimensions)."""
        
        if len(fixations) == 0:
            return np.zeros(24)
        
        features = []
        
        # Fixation count
        features.append(len(fixations))
        
        # Fixation durations
        durations = [(end - start) / self.sampling_rate for start, end in fixations]
        features.extend([
            np.sum(durations),           # Total duration
            np.mean(durations),          # Mean duration
            np.std(durations) if len(durations) > 1 else 0,  # Std duration
            np.max(durations),           # Max duration
            np.min(durations)            # Min duration
        ])
        
        # Fixation rate
        total_time = len(x) / self.sampling_rate
        features.append(len(fixations) / total_time)
        
        # Spatial dispersion (average distance from fixation centroid)
        dispersions = []
        for start, end in fixations:
            fix_x = x[start:end]
            fix_y = y[start:end]
            centroid_x = np.mean(fix_x)
            centroid_y = np.mean(fix_y)
            dispersion = np.mean(np.sqrt((fix_x - centroid_x)**2 + (fix_y - centroid_y)**2))
            dispersions.append(dispersion)
        
        features.extend([
            np.mean(dispersions),
            np.std(dispersions) if len(dispersions) > 1 else 0,
            np.max(dispersions),
            np.min(dispersions)
        ])
        
        # Fixation distribution entropy
        # Divide screen into grid and compute entropy of fixation distribution
        x_bins = np.histogram(x, bins=10, range=(0, 1920))[0]
        y_bins = np.histogram(y, bins=10, range=(0, 1080))[0]
        x_entropy = entropy(x_bins + 1e-10)  # Add small constant to avoid log(0)
        y_entropy = entropy(y_bins + 1e-10)
        features.extend([x_entropy, y_entropy])
        
        # Additional fixation statistics
        features.extend([
            np.median(durations),
            np.percentile(durations, 25),
            np.percentile(durations, 75),
            np.var(durations) if len(durations) > 1 else 0
        ])
        
        # Pad or truncate to 24 dimensions
        features = features[:24]
        while len(features) < 24:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_saccade_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        saccades: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Extract saccade-related features (30 dimensions)."""
        
        if len(saccades) == 0:
            return np.zeros(30)
        
        features = []
        
        # Saccade count
        features.append(len(saccades))
        
        # Saccade amplitudes and velocities
        amplitudes = []
        velocities = []
        
        for start, end in saccades:
            # Amplitude (distance traveled)
            dx = x[end-1] - x[start]
            dy = y[end-1] - y[start]
            amplitude = np.sqrt(dx**2 + dy**2)
            amplitudes.append(amplitude)
            
            # Peak velocity
            segment_velocity = self.compute_velocity(x[start:end], y[start:end])
            peak_velocity = np.max(segment_velocity)
            velocities.append(peak_velocity)
        
        # Amplitude statistics
        features.extend([
            np.mean(amplitudes),
            np.std(amplitudes) if len(amplitudes) > 1 else 0,
            np.max(amplitudes),
            np.min(amplitudes),
            np.median(amplitudes)
        ])
        
        # Velocity statistics
        features.extend([
            np.mean(velocities),
            np.std(velocities) if len(velocities) > 1 else 0,
            np.max(velocities),
            np.min(velocities),
            np.median(velocities)
        ])
        
        # Saccade rate
        total_time = len(x) / self.sampling_rate
        features.append(len(saccades) / total_time)
        
        # Saccade direction entropy
        directions = []
        for start, end in saccades:
            dx = x[end-1] - x[start]
            dy = y[end-1] - y[start]
            direction = np.arctan2(dy, dx)
            directions.append(direction)
        
        # Discretize directions into 8 bins
        direction_bins = np.histogram(directions, bins=8, range=(-np.pi, np.pi))[0]
        direction_entropy = entropy(direction_bins + 1e-10)
        features.append(direction_entropy)
        
        # Saccade latency (time between consecutive saccades)
        if len(saccades) > 1:
            latencies = [(saccades[i+1][0] - saccades[i][1]) / self.sampling_rate 
                        for i in range(len(saccades)-1)]
            features.extend([
                np.mean(latencies),
                np.std(latencies) if len(latencies) > 1 else 0,
                np.median(latencies)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Additional statistics
        features.extend([
            np.percentile(amplitudes, 25),
            np.percentile(amplitudes, 75),
            np.percentile(velocities, 25),
            np.percentile(velocities, 75),
            np.var(amplitudes) if len(amplitudes) > 1 else 0,
            np.var(velocities) if len(velocities) > 1 else 0
        ])
        
        # Pad or truncate to 30 dimensions
        features = features[:30]
        while len(features) < 30:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_pupil_features(
        self,
        pupil_left: np.ndarray,
        pupil_right: np.ndarray
    ) -> np.ndarray:
        """Extract pupil-related features (20 dimensions)."""
        
        # Handle missing data
        pupil_left_clean = pupil_left[~np.isnan(pupil_left) & (pupil_left > 0)]
        pupil_right_clean = pupil_right[~np.isnan(pupil_right) & (pupil_right > 0)]
        
        if len(pupil_left_clean) == 0 or len(pupil_right_clean) == 0:
            return np.zeros(20)
        
        features = []
        
        # Average pupil diameter
        pupil_avg = (pupil_left_clean + pupil_right_clean) / 2
        
        # Mean pupil diameter
        features.extend([
            np.mean(pupil_left_clean),
            np.mean(pupil_right_clean),
            np.mean(pupil_avg)
        ])
        
        # Pupil diameter variability
        features.extend([
            np.std(pupil_left_clean),
            np.std(pupil_right_clean),
            np.std(pupil_avg)
        ])
        
        # Pupil diameter range
        features.extend([
            np.max(pupil_left_clean) - np.min(pupil_left_clean),
            np.max(pupil_right_clean) - np.min(pupil_right_clean),
            np.max(pupil_avg) - np.min(pupil_avg)
        ])
        
        # Pupil diameter slope (change over time - indicates cognitive load)
        if len(pupil_avg) > 1:
            time = np.arange(len(pupil_avg))
            slope = np.polyfit(time, pupil_avg, 1)[0]
            features.append(slope)
        else:
            features.append(0)
        
        # Cross-correlation between eyes
        if len(pupil_left_clean) > 1 and len(pupil_right_clean) > 1:
            # Ensure same length
            min_len = min(len(pupil_left_clean), len(pupil_right_clean))
            corr = np.corrcoef(pupil_left_clean[:min_len], pupil_right_clean[:min_len])[0, 1]
            features.append(corr)
        else:
            features.append(0)
        
        # Binocular disparity
        pupil_diff = np.abs(pupil_left_clean - pupil_right_clean[:len(pupil_left_clean)])
        features.extend([
            np.mean(pupil_diff),
            np.std(pupil_diff),
            np.max(pupil_diff)
        ])
        
        # Additional statistics
        features.extend([
            np.median(pupil_avg),
            np.percentile(pupil_avg, 25),
            np.percentile(pupil_avg, 75),
            np.var(pupil_avg)
        ])
        
        # Pad or truncate to 20 dimensions
        features = features[:20]
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_blink_features(
        self,
        blinks: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Extract blink-related features (22 dimensions)."""
        
        if len(blinks) == 0:
            return np.zeros(22)
        
        features = []
        
        # Blink count
        features.append(len(blinks))
        
        # Blink rate
        total_time = self.window_samples / self.sampling_rate
        features.append(len(blinks) / total_time)
        
        # Blink durations
        durations = [(end - start) / self.sampling_rate for start, end in blinks]
        features.extend([
            np.mean(durations),
            np.std(durations) if len(durations) > 1 else 0,
            np.max(durations),
            np.min(durations),
            np.median(durations)
        ])
        
        # Inter-blink intervals
        if len(blinks) > 1:
            intervals = [(blinks[i+1][0] - blinks[i][1]) / self.sampling_rate 
                        for i in range(len(blinks)-1)]
            features.extend([
                np.mean(intervals),
                np.std(intervals) if len(intervals) > 1 else 0,
                np.median(intervals),
                np.min(intervals),
                np.max(intervals)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Blink amplitude (if available - placeholder)
        features.extend([0] * 10)  # Placeholder for additional blink metrics
        
        # Pad or truncate to 22 dimensions
        features = features[:22]
        while len(features) < 22:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_features_from_window(
        self,
        x_left: np.ndarray,
        y_left: np.ndarray,
        x_right: np.ndarray,
        y_right: np.ndarray,
        pupil_left: np.ndarray,
        pupil_right: np.ndarray
    ) -> np.ndarray:
        """
        Extract all features from a time window (96 dimensions).
        
        Args:
            x_left, y_left: Left eye gaze coordinates
            x_right, y_right: Right eye gaze coordinates
            pupil_left, pupil_right: Pupil diameters
        
        Returns:
            Feature vector (96 dimensions)
        """
        # Use binocular average for fixation/saccade detection
        x_avg = (x_left + x_right) / 2
        y_avg = (y_left + y_right) / 2
        
        # Detect events
        fixations = self.detect_fixations(x_avg, y_avg)
        saccades = self.detect_saccades(x_avg, y_avg, fixations)
        blinks = self.detect_blinks(pupil_left, pupil_right)
        
        # Extract features
        fixation_features = self.extract_fixation_features(x_avg, y_avg, fixations)  # 24 dims
        saccade_features = self.extract_saccade_features(x_avg, y_avg, saccades)    # 30 dims
        pupil_features = self.extract_pupil_features(pupil_left, pupil_right)       # 20 dims
        blink_features = self.extract_blink_features(blinks)                        # 22 dims
        
        # Concatenate all features
        features = np.concatenate([
            fixation_features,
            saccade_features,
            pupil_features,
            blink_features
        ])
        
        return features
    
    def preprocess(
        self,
        x_left: np.ndarray,
        y_left: np.ndarray,
        x_right: np.ndarray,
        y_right: np.ndarray,
        pupil_left: np.ndarray,
        pupil_right: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for eye-tracking data.
        
        Args:
            x_left, y_left: Left eye gaze coordinates (n_samples,)
            x_right, y_right: Right eye gaze coordinates (n_samples,)
            pupil_left, pupil_right: Pupil diameters (n_samples,)
        
        Returns:
            Dictionary containing extracted features
        """
        n_samples = len(x_left)
        n_windows = n_samples // self.window_samples
        
        # Truncate to fit whole windows
        n_samples_truncated = n_windows * self.window_samples
        
        # Extract features for each window
        features = np.zeros((n_windows, 96))
        
        for i in range(n_windows):
            start_idx = i * self.window_samples
            end_idx = start_idx + self.window_samples
            
            window_features = self.extract_features_from_window(
                x_left[start_idx:end_idx],
                y_left[start_idx:end_idx],
                x_right[start_idx:end_idx],
                y_right[start_idx:end_idx],
                pupil_left[start_idx:end_idx],
                pupil_right[start_idx:end_idx]
            )
            
            features[i] = window_features
        
        return {
            'features': features,
            'n_windows': n_windows
        }


if __name__ == "__main__":
    # Test preprocessing
    sampling_rate = 60
    duration = 60  # seconds
    n_samples = sampling_rate * duration
    
    # Simulate eye-tracking data
    t = np.arange(n_samples) / sampling_rate
    
    # Gaze coordinates with saccades and fixations
    x_left = 960 + 200 * np.sin(0.5 * t) + 50 * np.random.randn(n_samples)
    y_left = 540 + 200 * np.cos(0.5 * t) + 50 * np.random.randn(n_samples)
    x_right = x_left + 10 * np.random.randn(n_samples)
    y_right = y_left + 10 * np.random.randn(n_samples)
    
    # Pupil diameter with variations
    pupil_left = 4.0 + 0.5 * np.sin(0.1 * t) + 0.2 * np.random.randn(n_samples)
    pupil_right = pupil_left + 0.1 * np.random.randn(n_samples)
    
    # Add some blinks (missing data)
    blink_indices = np.random.choice(n_samples, size=20, replace=False)
    for idx in blink_indices:
        pupil_left[idx:idx+3] = np.nan
        pupil_right[idx:idx+3] = np.nan
    
    # Preprocess
    preprocessor = EyeTrackingPreprocessor()
    results = preprocessor.preprocess(
        x_left, y_left, x_right, y_right,
        pupil_left, pupil_right
    )
    
    print(f"Features shape: {results['features'].shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {results['features'].mean():.3f}")
    print(f"  Std: {results['features'].std():.3f}")
    print(f"  Min: {results['features'].min():.3f}")
    print(f"  Max: {results['features'].max():.3f}")

"""
Behavioral Data Preprocessing Pipeline

Implements feature extraction from behavioral task performance:
1. Reaction time statistics
2. Accuracy metrics  
3. Performance trend features
"""

import numpy as np
from scipy import signal
from scipy.stats import linregress
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BehavioralPreprocessor:
    """Behavioral data preprocessing and feature extraction."""
    
    def __init__(
        self,
        window_length: float = 1.0,  # seconds
        rt_min: float = 0.15,  # Minimum valid RT
        rt_max: float = 2.0    # Maximum valid RT
    ):
        self.window_length = window_length
        self.rt_min = rt_min
        self.rt_max = rt_max
    
    def clean_reaction_times(self, rts: np.ndarray) -> np.ndarray:
        """Remove outlier reaction times."""
        valid_mask = (rts >= self.rt_min) & (rts <= self.rt_max)
        return rts[valid_mask]
    
    def extract_rt_features(
        self,
        rts: np.ndarray,
        accuracy: np.ndarray
    ) -> np.ndarray:
        """Extract reaction time features (24 dimensions)."""
        
        if len(rts) == 0:
            return np.zeros(24)
        
        features = []
        
        # Overall RT statistics
        features.extend([
            np.mean(rts),
            np.median(rts),
            np.std(rts),
            np.min(rts),
            np.max(rts),
            np.max(rts) - np.min(rts),
            np.std(rts) / (np.mean(rts) + 1e-10),  # Coefficient of variation
            np.var(rts)
        ])
        
        # Correct vs incorrect trial RTs
        if len(accuracy) == len(rts):
            correct_rts = rts[accuracy == 1]
            incorrect_rts = rts[accuracy == 0]
            
            if len(correct_rts) > 0:
                features.extend([
                    np.mean(correct_rts),
                    np.std(correct_rts),
                    np.median(correct_rts)
                ])
            else:
                features.extend([0, 0, 0])
            
            if len(incorrect_rts) > 0:
                features.extend([
                    np.mean(incorrect_rts),
                    np.std(incorrect_rts),
                    np.median(incorrect_rts)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0] * 6)
        
        # Percentiles
        features.extend([
            np.percentile(rts, 25),
            np.percentile(rts, 75),
            np.percentile(rts, 90)
        ])
        
        # Skewness and kurtosis
        from scipy.stats import skew, kurtosis
        features.extend([
            skew(rts),
            kurtosis(rts)
        ])
        
        # Additional statistics
        features.extend([0] * (24 - len(features)))  # Pad to 24
        
        return np.array(features[:24])
    
    def extract_accuracy_features(
        self,
        accuracy: np.ndarray,
        trial_types: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract accuracy features (20 dimensions)."""
        
        if len(accuracy) == 0:
            return np.zeros(20)
        
        features = []
        
        # Overall accuracy
        features.append(np.mean(accuracy))
        
        # Hit rate and false alarm rate (if trial types available)
        if trial_types is not None:
            go_trials = trial_types == 1
            nogo_trials = trial_types == 0
            
            if np.sum(go_trials) > 0:
                hit_rate = np.mean(accuracy[go_trials])
                features.append(hit_rate)
            else:
                features.append(0)
            
            if np.sum(nogo_trials) > 0:
                correct_reject = np.mean(accuracy[nogo_trials])
                false_alarm_rate = 1 - correct_reject
                features.append(false_alarm_rate)
            else:
                features.append(0)
            
            # d-prime and beta (signal detection theory)
            if np.sum(go_trials) > 0 and np.sum(nogo_trials) > 0:
                from scipy.stats import norm
                hit = np.clip(hit_rate, 0.01, 0.99)
                fa = np.clip(false_alarm_rate, 0.01, 0.99)
                dprime = norm.ppf(hit) - norm.ppf(fa)
                beta = np.exp((norm.ppf(fa)**2 - norm.ppf(hit)**2) / 2)
                features.extend([dprime, beta])
            else:
                features.extend([0, 0])
        else:
            features.extend([0] * 4)
        
        # Error types
        omission_errors = np.sum(accuracy == 0)
        commission_errors = np.sum(accuracy == 0)
        features.extend([
            omission_errors / len(accuracy),
            commission_errors / len(accuracy)
        ])
        
        # Performance variability
        # Moving window accuracy
        window_size = 10
        if len(accuracy) >= window_size:
            moving_acc = [np.mean(accuracy[i:i+window_size]) 
                         for i in range(len(accuracy) - window_size + 1)]
            features.extend([
                np.mean(moving_acc),
                np.std(moving_acc),
                np.min(moving_acc),
                np.max(moving_acc)
            ])
        else:
            features.extend([0] * 4)
        
        # Additional metrics
        features.extend([0] * (20 - len(features)))  # Pad to 20
        
        return np.array(features[:20])
    
    def extract_performance_trend_features(
        self,
        rts: np.ndarray,
        accuracy: np.ndarray
    ) -> np.ndarray:
        """Extract performance trend features (20 dimensions)."""
        
        if len(rts) < 2 or len(accuracy) < 2:
            return np.zeros(20)
        
        features = []
        
        # Learning curve (RT trend)
        trial_nums = np.arange(len(rts))
        if len(rts) > 1:
            slope, intercept, r_value, _, _ = linregress(trial_nums, rts)
            features.extend([slope, r_value**2])
        else:
            features.extend([0, 0])
        
        # Accuracy trend
        if len(accuracy) > 1:
            slope_acc, _, r_acc, _, _ = linregress(trial_nums, accuracy)
            features.extend([slope_acc, r_acc**2])
        else:
            features.extend([0, 0])
        
        # Speed-accuracy tradeoff
        if len(rts) == len(accuracy):
            correlation = np.corrcoef(rts, accuracy)[0, 1]
            features.append(correlation)
        else:
            features.append(0)
        
        # Post-error slowing
        if len(accuracy) > 1 and len(rts) > 1:
            error_indices = np.where(accuracy == 0)[0]
            post_error_rts = []
            for idx in error_indices:
                if idx + 1 < len(rts):
                    post_error_rts.append(rts[idx + 1])
            
            if len(post_error_rts) > 0:
                avg_post_error_rt = np.mean(post_error_rts)
                avg_rt = np.mean(rts)
                post_error_slowing = (avg_post_error_rt - avg_rt) / avg_rt
                features.append(post_error_slowing)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # Performance decrement (fatigue effect)
        if len(rts) >= 20:
            first_half_rt = np.mean(rts[:len(rts)//2])
            second_half_rt = np.mean(rts[len(rts)//2:])
            rt_decrement = (second_half_rt - first_half_rt) / first_half_rt
            
            first_half_acc = np.mean(accuracy[:len(accuracy)//2])
            second_half_acc = np.mean(accuracy[len(accuracy)//2:])
            acc_decrement = (first_half_acc - second_half_acc)
            
            features.extend([rt_decrement, acc_decrement])
        else:
            features.extend([0, 0])
        
        # Additional metrics
        features.extend([0] * (20 - len(features)))  # Pad to 20
        
        return np.array(features[:20])
    
    def preprocess(
        self,
        reaction_times: np.ndarray,
        accuracy: np.ndarray,
        trial_types: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for behavioral data.
        
        Returns:
            Dictionary containing extracted features (64 dimensions per window)
        """
        # Clean reaction times
        rts_clean = self.clean_reaction_times(reaction_times)
        
        # Align accuracy with cleaned RTs if needed
        if len(rts_clean) < len(accuracy):
            valid_mask = (reaction_times >= self.rt_min) & (reaction_times <= self.rt_max)
            accuracy = accuracy[valid_mask]
            if trial_types is not None:
                trial_types = trial_types[valid_mask]
        
        # Extract features
        rt_features = self.extract_rt_features(rts_clean, accuracy)  # 24 dims
        acc_features = self.extract_accuracy_features(accuracy, trial_types)  # 20 dims
        trend_features = self.extract_performance_trend_features(rts_clean, accuracy)  # 20 dims
        
        # Concatenate
        features = np.concatenate([rt_features, acc_features, trend_features])
        
        return {
            'features': features.reshape(1, -1),  # (1, 64)
            'cleaned_rts': rts_clean,
            'accuracy': accuracy
        }

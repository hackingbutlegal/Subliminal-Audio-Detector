import numpy as np
import pprint
from scipy.signal import hilbert, correlate, welch
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class SubliminalAudioDetector:
    def __init__(self, chunk_duration=10, overlap=1):
        self.detectors = []
        self.initialize_detectors()
        self.chunk_duration = chunk_duration  # Duration of each chunk in seconds
        self.overlap = overlap  # Overlap between chunks in seconds
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('SubliminalAudioDetector')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_detectors(self):
        self.detectors = [
            self.detect_backmasking,
            self.detect_high_frequency_content,
            self.detect_amplitude_modulation,
            self.detect_stereo_phase_manipulation,
            self.detect_binaural_beats
        ]


    def load_audio(self, audio_file):
        self.logger.info(f"Loading audio file: {audio_file}")
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
        return audio, sr

    def aggregate_results(self, results):
        aggregated = {}
        for detector_name, chunk_results in results.items():
            if not chunk_results:
                continue

            if detector_name == 'detect_backmasking':
                aggregated[detector_name] = self.aggregate_backmasking(
                    chunk_results)
            elif detector_name == 'detect_high_frequency_content':
                aggregated[detector_name] = self.aggregate_high_frequency(
                    chunk_results)
            elif detector_name == 'detect_amplitude_modulation':
                aggregated[detector_name] = self.aggregate_amplitude_modulation(
                    chunk_results)
            elif detector_name == 'detect_stereo_phase_manipulation':
                aggregated[detector_name] = self.aggregate_stereo_phase(
                    chunk_results)
            elif detector_name == 'detect_binaural_beats':
                aggregated[detector_name] = self.aggregate_binaural_beats(
                    chunk_results)

        return aggregated

    def aggregate_backmasking(self, chunk_results):
        detected = any(result['backmasking_detected']
                       for result in chunk_results)
        confidence = max(result['confidence'] for result in chunk_results)
        locations = [
            loc for result in chunk_results for loc in result['potential_locations']]
        return {
            "backmasking_detected": detected,
            "confidence": confidence,
            "potential_locations": locations
        }

    def aggregate_high_frequency(self, chunk_results):
        percentages = [result['ultrasonic_power_percentage']
            for result in chunk_results]
        avg_percentage = np.mean(percentages)
        if avg_percentage > 5:
            level = "High"
        elif avg_percentage > 1:
            level = "Moderate"
        else:
            level = "Low"
        return {
            "high_frequency_content_detected": level,
            "ultrasonic_power_percentage": avg_percentage,
            "frequency_range": chunk_results[0]['frequency_range']
        }

    def aggregate_amplitude_modulation(self, chunk_results):
        detected = any(result['amplitude_modulation'] ==
                       "Detected" for result in chunk_results)
        frequencies = [result['dominant_frequency']
            for result in chunk_results]
        depths = [result['modulation_depth'] for result in chunk_results]
        return {
            "amplitude_modulation": "Detected" if detected else "Not Detected",
            "dominant_frequency": np.mean(frequencies),
            "modulation_depth": np.mean(depths)
        }

    def aggregate_stereo_phase(self, chunk_results):
        detected = any(result['stereo_phase_manipulation']
                       == "Detected" for result in chunk_results)
        phase_diffs = [result['mean_phase_difference']
            for result in chunk_results]
        correlations = [result['correlation'] for result in chunk_results]
        return {
            "stereo_phase_manipulation": "Detected" if detected else "Not Detected",
            "mean_phase_difference": np.mean(phase_diffs),
            "correlation": np.mean(correlations)
        }

    def aggregate_binaural_beats(self, chunk_results):
        detected = any(result['binaural_beats'] ==
                       "Detected" for result in chunk_results)
        frequencies = [result['beat_frequency']
            for result in chunk_results if result['beat_frequency'] is not None]
        confidences = [result['confidence']
            for result in chunk_results if result['confidence'] is not None]
        return {
            "binaural_beats": "Detected" if detected else "Not Detected",
            "beat_frequency": np.mean(frequencies) if frequencies else None,
            "confidence": np.mean(confidences) if confidences else None
        }

    def detect_backmasking(self, audio, sr, start_time):
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Reverse the audio
        audio_reversed = audio[::-1]
        
        # Compute the cross-correlation
        correlation = correlate(audio, audio_reversed, mode='same')
        
        # Normalize the correlation
        correlation /= np.max(np.abs(correlation))
        
        # Define threshold for significant correlation
        threshold = 0.7
        
        # Find peaks in correlation above threshold
        peaks = np.where(correlation > threshold)[0]
        
        # Convert peak positions to time
        peak_times = peaks / sr + start_time  # Add start_time to align with chunk position
        
        # Analyze the results
        if len(peak_times) > 0:
            return {
                "detector": "detect_backmasking",
                "backmasking_detected": True,
                "confidence": np.max(correlation[peaks]),
                "potential_locations": peak_times.tolist(),
                "timestamp": start_time
            }
        else:
            return {
                "detector": "detect_backmasking",
                "backmasking_detected": False,
                "confidence": 0,
                "potential_locations": [],
                "timestamp": start_time
            }

    def detect_high_frequency_content(self, audio, sr, start_time):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        human_hearing_limit = 20000
        ultrasonic_low = 20000
        ultrasonic_high = sr // 2

        frequencies, psd = welch(audio, sr, nperseg=1024)

        total_power = np.sum(psd)
        ultrasonic_mask = (frequencies >= ultrasonic_low) & (frequencies <= ultrasonic_high)
        ultrasonic_power = np.sum(psd[ultrasonic_mask])

        ultrasonic_percentage = (ultrasonic_power / total_power) * 100 if total_power > 0 else 0

        if ultrasonic_percentage > 10:
            level = "Very High"
        elif ultrasonic_percentage > 5:
            level = "High"
        elif ultrasonic_percentage > 1:
            level = "Moderate"
        elif ultrasonic_percentage > 0.1:
            level = "Low"
        else:
            level = "Negligible"

        return {
            "detector": "detect_high_frequency_content",
            "high_frequency_content_detected": level,
            "ultrasonic_power_percentage": ultrasonic_percentage,
            "frequency_range": f"{ultrasonic_low}-{ultrasonic_high} Hz",
            "timestamp": start_time
        }

    def detect_amplitude_modulation(self, audio, sr, start_time):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        analytic_signal = hilbert(audio)
        amplitude_envelope = np.abs(analytic_signal)
        
        modulation = np.diff(amplitude_envelope)
        modulation_fft = np.fft.fft(modulation)
        modulation_freqs = np.fft.fftfreq(len(modulation), 1/sr)
        
        positive_freqs = modulation_freqs[:len(modulation_freqs)//2]
        positive_modulation_fft = np.abs(modulation_fft[:len(modulation_freqs)//2])
        
        dominant_freq_idx = np.argmax(positive_modulation_fft)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        modulation_depth = (np.max(amplitude_envelope) - np.min(amplitude_envelope)) / np.max(amplitude_envelope) if np.max(amplitude_envelope) > 0 else 0
        
        freq_threshold_low = 0.5
        freq_threshold_high = 40
        depth_threshold_low = 0.05
        depth_threshold_high = 0.2
        
        if freq_threshold_low <= dominant_freq <= freq_threshold_high:
            if modulation_depth > depth_threshold_high:
                detection = "Strong"
                confidence = 0.9
            elif modulation_depth > depth_threshold_low:
                detection = "Moderate"
                confidence = 0.7
            else:
                detection = "Weak"
                confidence = 0.5
        else:
            detection = "Not Detected"
            confidence = 0.3
        
        return {
            "detector": "detect_amplitude_modulation",
            "amplitude_modulation": detection,
            "dominant_frequency": dominant_freq,
            "modulation_depth": modulation_depth,
            "confidence": confidence,
            "timestamp": start_time
        }

    def detect_stereo_phase_manipulation(self, audio, sr, start_time):
        if audio.ndim == 1 or audio.shape[0] == 1:
            return {
                "detector": "detect_stereo_phase_manipulation",
                "stereo_phase_manipulation": "Not applicable (mono audio)",
                "mean_phase_difference": None,
                "correlation": None,
                "confidence": 1.0,
                "timestamp": start_time
            }

        left_channel = audio[0]
        right_channel = audio[1]

        left_analytic = hilbert(left_channel)
        right_analytic = hilbert(right_channel)

        phase_diff = np.angle(left_analytic) - np.angle(right_analytic)
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

        mean_phase_diff = np.mean(np.abs(phase_diff))
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]

        phase_threshold = 0.5
        correlation_threshold = 0.95

        if mean_phase_diff > phase_threshold and correlation < correlation_threshold:
            detection = "Detected"
            confidence = 0.8 + (mean_phase_diff - phase_threshold) * 0.2
        else:
            detection = "Not Detected"
            confidence = 0.2 + (correlation - correlation_threshold) * 0.8

        confidence = min(max(confidence, 0), 1)  # Ensure confidence is between 0 and 1

        return {
            "detector": "detect_stereo_phase_manipulation",
            "stereo_phase_manipulation": detection,
            "mean_phase_difference": mean_phase_diff,
            "correlation": correlation,
            "confidence": confidence,
            "timestamp": start_time
        }

    def detect_binaural_beats(self, audio, sr, start_time):
        if audio.ndim == 1 or audio.shape[0] == 1:
            return {
                "detector": "detect_binaural_beats",
                "binaural_beats": "Not applicable (mono audio)",
                "beat_frequency": None,
                "confidence": 1.0,
                "timestamp": start_time
            }

        left_channel = audio[0]
        right_channel = audio[1]

        n_fft = 2048
        hop_length = 512

        left_stft = librosa.stft(left_channel, n_fft=n_fft, hop_length=hop_length)
        right_stft = librosa.stft(right_channel, n_fft=n_fft, hop_length=hop_length)

        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        phase_diff = np.angle(left_stft) - np.angle(right_stft)

        freq_diff = np.diff(np.unwrap(phase_diff, axis=0), axis=0) / (2 * np.pi * hop_length / sr)

        mean_freq_diff = np.mean(np.abs(freq_diff), axis=1)

        max_diff_bin = np.argmax(mean_freq_diff)
        beat_frequency = frequencies[max_diff_bin]

        confidence = mean_freq_diff[max_diff_bin] / np.mean(mean_freq_diff) if np.mean(mean_freq_diff) > 0 else 0

        freq_threshold_low = 1
        freq_threshold_high = 30
        confidence_threshold = 2

        if freq_threshold_low <= beat_frequency <= freq_threshold_high and confidence > confidence_threshold:
            detection = "Detected"
            confidence = min(confidence / 4, 1)  # Normalize confidence to 0-1 range
        else:
            detection = "Not Detected"
            confidence = max(1 - confidence / 4, 0)  # Inverse and normalize confidence to 0-1 range

        return {
            "detector": "detect_binaural_beats",
            "binaural_beats": detection,
            "beat_frequency": beat_frequency,
            "confidence": confidence,
            "timestamp": start_time
        }

    def analyze_audio(self, audio_file):
        self.logger.info(f"Analyzing audio file: {audio_file}")
        try:
            audio, sr = self.load_audio(audio_file)
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            return f"Error: Unable to load audio file. {str(e)}"

        chunk_size = int(self.chunk_duration * sr)
        hop_length = chunk_size - int(self.overlap * sr)

        results = {detector.__name__: [] for detector in self.detectors}

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, audio.shape[-1], hop_length):
                chunk = audio[..., i:i+chunk_size]
                if chunk.shape[-1] < chunk_size:
                    pad_width = [(0, 0)] * (chunk.ndim - 1) + [(0, chunk_size - chunk.shape[-1])]
                    chunk = np.pad(chunk, pad_width, mode='constant')

                start_time = i / sr  # Calculate start time for this chunk

                for detector in self.detectors:
                    futures.append(executor.submit(detector, chunk, sr, start_time))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results[result['detector']].append(result)
                except Exception as e:
                    self.logger.error(f"Error processing chunk: {e}")

        try:
            aggregated_results = self.aggregate_results(results)
            report = self.generate_report(aggregated_results)
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            report = f"Error: Unable to generate report. {str(e)}"

        return report


    def generate_report(self, results):
        report = "# Subliminal Audio Detection Report\n\n"

        for detector, detector_results in results.items():
            report += f"## {detector.replace('_', ' ').title()}\n\n"
            
            if isinstance(detector_results, dict):
                # Handle single result
                report += self.format_result(detector_results, detector)
            elif isinstance(detector_results, list):
                # Handle multiple results
                for result in detector_results:
                    report += self.format_result(result, detector)
            elif isinstance(detector_results, str):
                # Handle unexpected string result
                report += f"Unexpected result format for {detector}: {detector_results}\n\n"
            else:
                # Handle any other unexpected type
                report += f"Unexpected result type for {detector}: {type(detector_results)}\n\n"

        report += "## Overall Conclusion\n\n"
        detected_techniques = [key for key, value in results.items() if self.is_technique_detected(value)]
        if detected_techniques:
            report += "The audio file shows signs of: " + ", ".join(detected_techniques) + ". "
            report += "These detected elements could be for creative effects or potentially for more subtle purposes such as subliminal messaging. "
            report += "Further investigation may be warranted to determine their purpose and potential effect on the listener.\n\n"
        else:
            report += "The audio file does not contain significant elements typically associated with subliminal audio techniques. "
            report += "No further investigation appears necessary based on this analysis.\n\n"

        report += "**Note**: This analysis provides an automated assessment of potential subliminal audio techniques. "
        report += "The presence of these elements does not necessarily indicate intentional subliminal messaging. "
        report += "Conversely, the absence of detected elements does not guarantee the lack of subliminal content. "
        report += "Human review and context consideration are recommended for a comprehensive evaluation.\n\n"

        return report

    def format_result(self, result, detector):
        formatted = ""
        if not isinstance(result, dict):
            return f"Unexpected result format for {detector}: {result}\n\n"

        detection_key = detector.split('_')[-1]
        formatted += f"- **Detection**: {result.get(detection_key, 'N/A')}\n"
        formatted += f"- **Confidence**: {result.get('confidence', 'N/A')}\n"
        formatted += f"- **Timestamp**: {result.get('timestamp', 'N/A')} seconds\n"
        
        for key, value in result.items():
            if key not in ['detector', detection_key, 'confidence', 'timestamp']:
                if isinstance(value, float):
                    formatted += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
                else:
                    formatted += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        formatted += "\n**Interpretation**: "
        if detector == 'detect_high_frequency_content':
            formatted += self.interpret_high_frequency(result)
        elif detector == 'detect_amplitude_modulation':
            formatted += self.interpret_amplitude_modulation(result)
        elif detector == 'detect_stereo_phase_manipulation':
            formatted += self.interpret_stereo_phase(result)
        elif detector == 'detect_binaural_beats':
            formatted += self.interpret_binaural_beats(result)
        else:
            formatted += "No specific interpretation available for this detection method."
        
        formatted += "\n\n"
        return formatted

    def is_technique_detected(self, value):
        if isinstance(value, dict):
            detection_key = next(iter(value)).split('_')[-1]
            return value.get(detection_key, 'Not Detected') not in ['Not Detected', 'Not applicable (mono audio)']
        elif isinstance(value, list):
            return any(self.is_technique_detected(item) for item in value)
        elif isinstance(value, str):
            return value not in ['Not Detected', 'Not applicable (mono audio)']
        return False


    def interpret_high_frequency(self, result):
        level = result['high_frequency_content_detected']
        if level == 'Negligible':
            return "The audio file contains negligible high-frequency content in the ultrasonic range. There is no significant ultrasonic content that could be used for subliminal messaging."
        elif level == 'Low':
            return "The audio file contains low levels of ultrasonic content. While present, it's unlikely to be significant enough for effective subliminal messaging."
        elif level == 'Moderate':
            return "The audio file contains moderate levels of ultrasonic content. This could potentially be used for subliminal messaging, but its effectiveness would depend on various factors."
        elif level in ['High', 'Very High']:
            return f"The audio file contains {level.lower()} levels of ultrasonic content. This might be used for subliminal messaging or could be an artifact of the recording process. Further investigation is recommended."
        else:
            return "Unexpected result in high frequency content detection. Further investigation is needed."


    def interpret_amplitude_modulation(self, result):
        detection = result['amplitude_modulation']
        freq = result['dominant_frequency']
        if detection == 'Strong':
            if 0.5 <= freq <= 40:
                return "Strong amplitude modulation was detected. The dominant frequency falls within the range typically associated with subliminal effects (0.5-40 Hz). This could potentially be used for subliminal messaging with high effectiveness."
            else:
                return "Strong amplitude modulation was detected. However, the dominant frequency is outside the range typically associated with subliminal effects (0.5-40 Hz). This modulation is likely for other audio effects or could be an unintentional artifact."
        elif detection == 'Moderate':
            if 0.5 <= freq <= 40:
                return "Moderate amplitude modulation was detected. The dominant frequency falls within the range typically associated with subliminal effects (0.5-40 Hz). This could potentially be used for subliminal messaging, but its effectiveness might be limited."
            else:
                return "Moderate amplitude modulation was detected. However, the dominant frequency is outside the range typically associated with subliminal effects (0.5-40 Hz). This modulation is likely for other audio effects or could be an unintentional artifact."
        elif detection == 'Weak':
            return "Weak amplitude modulation was detected. While present, it's unlikely to be significant enough for effective subliminal messaging."
        elif detection == 'Not Detected':
            return "No significant amplitude modulation was detected that meets the criteria for potential subliminal effects."
        else:
            return "Unexpected result in amplitude modulation detection. Further investigation is needed."


    def interpret_stereo_phase(self, result):
            if result['stereo_phase_manipulation'] == "Not applicable (mono audio)":
                return "Stereo phase manipulation detection is not applicable as the audio is mono."
            elif result['stereo_phase_manipulation'] == 'Detected':
                return f"The analysis detected potential stereo phase manipulation. The average phase difference of {result['mean_phase_difference']:.3f} radians is substantial. This could be for creative audio effects or potentially for more subtle purposes such as subliminal messaging."
            elif result['stereo_phase_manipulation'] == 'Not Detected':
                return "No significant stereo phase manipulation was detected. The stereo channels appear to be in normal phase relationship."
            else:
                return "Unexpected result in stereo phase manipulation detection. Further investigation is needed."

    def interpret_binaural_beats(self, result):
        if result['binaural_beats'] == "Not applicable (mono audio)":
            return "Binaural beats detection is not applicable as the audio is mono."
        elif result['binaural_beats'] == 'Detected':
            if 1 <= result['beat_frequency'] <= 30:
                return f"Potential binaural beats were detected. The beat frequency of {result['beat_frequency']:.2f} Hz falls within the range typically associated with binaural beats (1-30 Hz). This could potentially be used for influencing brainwave states or subliminal effects."
            else:
                return f"Potential binaural beats were detected. However, the beat frequency of {result['beat_frequency']:.2f} Hz is outside the range typically associated with binaural beats (1-30 Hz). This frequency difference might be for other audio effects or could be unintentional."
        elif result['binaural_beats'] == 'Not Detected':
            return "No significant binaural beats were detected in the frequency range typically associated with subliminal effects."
        else:
            return "Unexpected result in binaural beats detection. Further investigation is needed."


    def load_audio(self, audio_file):
        self.logger.info(f"Loading audio file: {audio_file}")
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
        return audio, sr

    def aggregate_results(self, results):
        aggregated = {}
        for detector_name, chunk_results in results.items():
            if not chunk_results:
                continue
            
            if detector_name == 'detect_backmasking':
                aggregated[detector_name] = self.aggregate_backmasking(chunk_results)
            elif detector_name == 'detect_high_frequency_content':
                aggregated[detector_name] = self.aggregate_high_frequency(chunk_results)
            elif detector_name == 'detect_amplitude_modulation':
                aggregated[detector_name] = self.aggregate_amplitude_modulation(chunk_results)
            elif detector_name == 'detect_stereo_phase_manipulation':
                aggregated[detector_name] = self.aggregate_stereo_phase(chunk_results)
            elif detector_name == 'detect_binaural_beats':
                aggregated[detector_name] = self.aggregate_binaural_beats(chunk_results)

        return aggregated

    def aggregate_backmasking(self, chunk_results):
        detected = any(result['backmasking_detected'] for result in chunk_results)
        confidence = max(result['confidence'] for result in chunk_results)
        locations = [loc for result in chunk_results for loc in result['potential_locations']]
        return {
            "backmasking_detected": detected,
            "confidence": confidence,
            "potential_locations": locations
        }

    def aggregate_high_frequency(self, chunk_results):
        percentages = [result['ultrasonic_power_percentage'] for result in chunk_results]
        avg_percentage = np.mean(percentages)
        if avg_percentage > 5:
            level = "High"
        elif avg_percentage > 1:
            level = "Moderate"
        else:
            level = "Low"
        return {
            "high_frequency_content_detected": level,
            "ultrasonic_power_percentage": avg_percentage,
            "frequency_range": chunk_results[0]['frequency_range']
        }

    def aggregate_amplitude_modulation(self, chunk_results):
        detected = any(result['amplitude_modulation'] == "Detected" for result in chunk_results)
        frequencies = [result['dominant_frequency'] for result in chunk_results]
        depths = [result['modulation_depth'] for result in chunk_results]
        return {
            "amplitude_modulation": "Detected" if detected else "Not Detected",
            "dominant_frequency": np.mean(frequencies),
            "modulation_depth": np.mean(depths)
        }

    def aggregate_stereo_phase(self, chunk_results):
        detected = any(result['stereo_phase_manipulation'] == "Detected" for result in chunk_results)
        phase_diffs = [result['mean_phase_difference'] for result in chunk_results]
        correlations = [result['correlation'] for result in chunk_results]
        return {
            "stereo_phase_manipulation": "Detected" if detected else "Not Detected",
            "mean_phase_difference": np.mean(phase_diffs),
            "correlation": np.mean(correlations)
        }

    def aggregate_binaural_beats(self, chunk_results):
        detected = any(result['binaural_beats'] == "Detected" for result in chunk_results)
        frequencies = [result['beat_frequency'] for result in chunk_results if result['beat_frequency'] is not None]
        confidences = [result['confidence'] for result in chunk_results if result['confidence'] is not None]
        return {
            "binaural_beats": "Detected" if detected else "Not Detected",
            "beat_frequency": np.mean(frequencies) if frequencies else None,
            "confidence": np.mean(confidences) if confidences else None
        }

    def detect_backmasking(self, audio, sr, start_time):
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Reverse the audio
        audio_reversed = audio[::-1]
        
        # Compute the cross-correlation
        correlation = correlate(audio, audio_reversed, mode='same')
        
        # Normalize the correlation
        correlation /= np.max(np.abs(correlation))
        
        # Define threshold for significant correlation
        threshold = 0.7
        
        # Find peaks in correlation above threshold
        peaks = np.where(correlation > threshold)[0]
        
        # Convert peak positions to time
        peak_times = peaks / sr
        
        # Analyze the results
        if len(peak_times) > 0:
            return {
                "detector": "detect_backmasking",
                "backmasking_detected": True,
                "confidence": np.max(correlation[peaks]),
                "potential_locations": peak_times.tolist()
            }
        else:
            return {
                "detector": "detect_backmasking",
                "backmasking_detected": False,
                "confidence": 0,
                "potential_locations": []
            }

    def detect_high_frequency_content(self, audio, sr, start_time):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        human_hearing_limit = 20000
        ultrasonic_low = 20000
        ultrasonic_high = sr // 2

        frequencies, psd = welch(audio, sr, nperseg=1024)

        total_power = np.sum(psd)
        ultrasonic_mask = (frequencies >= ultrasonic_low) & (frequencies <= ultrasonic_high)
        ultrasonic_power = np.sum(psd[ultrasonic_mask])

        ultrasonic_percentage = (ultrasonic_power / total_power) * 100 if total_power > 0 else 0

        # Adjust thresholds for more granular classification
        if ultrasonic_percentage > 10:
            level = "Very High"
        elif ultrasonic_percentage > 5:
            level = "High"
        elif ultrasonic_percentage > 1:
            level = "Moderate"
        elif ultrasonic_percentage > 0.1:
            level = "Low"
        else:
            level = "Negligible"

        return {
            "detector": "detect_high_frequency_content",
            "high_frequency_content_detected": level,
            "ultrasonic_power_percentage": ultrasonic_percentage,
            "frequency_range": f"{ultrasonic_low}-{ultrasonic_high} Hz"
        }

    def detect_amplitude_modulation(self, audio, sr, start_time):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        analytic_signal = hilbert(audio)
        amplitude_envelope = np.abs(analytic_signal)
        
        modulation = np.diff(amplitude_envelope)
        modulation_fft = np.fft.fft(modulation)
        modulation_freqs = np.fft.fftfreq(len(modulation), 1/sr)
        
        positive_freqs = modulation_freqs[:len(modulation_freqs)//2]
        positive_modulation_fft = np.abs(modulation_fft[:len(modulation_freqs)//2])
        
        dominant_freq_idx = np.argmax(positive_modulation_fft)
        dominant_freq = positive_freqs[dominant_freq_idx]
        
        modulation_depth = (np.max(amplitude_envelope) - np.min(amplitude_envelope)) / np.max(amplitude_envelope) if np.max(amplitude_envelope) > 0 else 0
        
        freq_threshold_low = 0.5
        freq_threshold_high = 40
        depth_threshold_low = 0.05
        depth_threshold_high = 0.2
        
        if freq_threshold_low <= dominant_freq <= freq_threshold_high:
            if modulation_depth > depth_threshold_high:
                detection = "Strong"
                confidence = 0.9
            elif modulation_depth > depth_threshold_low:
                detection = "Moderate"
                confidence = 0.7
            else:
                detection = "Weak"
                confidence = 0.5
        else:
            detection = "Not Detected"
            confidence = 0.3
        
        return {
            "detector": "detect_amplitude_modulation",
            "amplitude_modulation": detection,
            "dominant_frequency": dominant_freq,
            "modulation_depth": modulation_depth,
            "confidence": confidence,
            "timestamp": start_time
        }

    def detect_stereo_phase_manipulation(self, audio, sr, start_time):
        # Ensure the audio is stereo
        if audio.ndim == 1 or audio.shape[0] == 1:
            return {
                "detector": "detect_stereo_phase_manipulation",
                "stereo_phase_manipulation": "Not applicable (mono audio)",
                "mean_phase_difference": None,
                "correlation": None
            }

        left_channel = audio[0]
        right_channel = audio[1]

        # Compute the analytic signal using the Hilbert transform
        left_analytic = hilbert(left_channel)
        right_analytic = hilbert(right_channel)

        # Calculate the instantaneous phase difference
        phase_diff = np.angle(left_analytic) - np.angle(right_analytic)

        # Wrap the phase difference to the range [-pi, pi]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

        # Calculate the mean absolute phase difference
        mean_phase_diff = np.mean(np.abs(phase_diff))

        # Calculate the correlation between channels
        correlation = np.corrcoef(left_channel, right_channel)[0, 1]

        # Define thresholds for detection
        phase_threshold = 0.5  # radians
        correlation_threshold = 0.95

        # Analyze the results
        detection = "Detected" if (mean_phase_diff > phase_threshold and correlation < correlation_threshold) else "Not Detected"

        return {
            "detector": "detect_stereo_phase_manipulation",
            "stereo_phase_manipulation": detection,
            "mean_phase_difference": mean_phase_diff,
            "correlation": correlation
        }

    def detect_binaural_beats(self, audio, sr, start_time):
        # Ensure the audio is stereo
        if audio.ndim == 1 or audio.shape[0] == 1:
            return {
                "detector": "detect_binaural_beats",
                "binaural_beats": "Not applicable (mono audio)",
                "beat_frequency": None,
                "confidence": None,
                "timestamp": start_time
            }

        left_channel = audio[0]
        right_channel = audio[1]

        # Compute the Short-Time Fourier Transform (STFT) for both channels
        n_fft = 2048
        hop_length = 512
        
        # Corrected STFT computation
        left_stft = librosa.stft(left_channel, n_fft=n_fft, hop_length=hop_length)
        right_stft = librosa.stft(right_channel, n_fft=n_fft, hop_length=hop_length)

        # Get the frequency bins
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Compute the phase difference between channels
        phase_diff = np.angle(left_stft) - np.angle(right_stft)

        # Compute the instantaneous frequency difference
        freq_diff = np.diff(np.unwrap(phase_diff, axis=0), axis=0) / (2 * np.pi * hop_length / sr)

        # Compute the mean frequency difference across time
        mean_freq_diff = np.mean(np.abs(freq_diff), axis=1)

        # Find the frequency bin with the maximum mean difference
        max_diff_bin = np.argmax(mean_freq_diff)
        beat_frequency = frequencies[max_diff_bin]

        # Compute a confidence score based on the prominence of the beat frequency
        confidence = mean_freq_diff[max_diff_bin] / np.mean(mean_freq_diff) if np.mean(mean_freq_diff) > 0 else 0

        # Define thresholds for detection
        freq_threshold_low = 1  # 1 Hz
        freq_threshold_high = 30  # 30 Hz
        confidence_threshold = 2  # The beat frequency should be at least twice as prominent as the average

        # Analyze the results
        detection = "Detected" if (freq_threshold_low <= beat_frequency <= freq_threshold_high and confidence > confidence_threshold) else "Not Detected"

        return {
            "detector": "detect_binaural_beats",
            "binaural_beats": detection,
            "beat_frequency": beat_frequency,
            "confidence": confidence,
            "timestamp": start_time
        }

# Usage example
if __name__ == "__main__":
    detector = SubliminalAudioDetector(chunk_duration=10, overlap=1)
    report = detector.analyze_audio("/path/to/audio_file.wav")
    print(report)


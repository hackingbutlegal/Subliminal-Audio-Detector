# Subliminal Audio Detector

## Overview

The Subliminal Audio Detector is a Python script designed to analyze audio files for potential subliminal content using various detection techniques. It employs a modular, performance-optimized approach to process large audio files efficiently.

## Features

- Detection of multiple subliminal audio techniques:
  - Backmasking
  - High-frequency content
  - Amplitude modulation
  - Stereo phase manipulation
  - Binaural beats
- Chunked processing for efficient handling of large audio files
- Parallel processing for improved performance
- Configurable chunk size and overlap
- Comprehensive result aggregation and reporting

## Detection Techniques

1. **Backmasking Detection**
   Backmasking involves reversing audio to embed hidden messages that may be perceived subconsciously when played forward.
   - Looks for similarities between the forward and reversed audio.
   - Returns potential locations of backmasked content.

2. **High-Frequency Content Detection**
   This subliminal audio technique involves embedding ultrasonic frequencies (above 20 kHz) that are typically inaudible to humans but may be subconsciously perceived.
   - Analyzes the presence of ultrasonic frequencies.
   - Reports the percentage of power in the ultrasonic range.

3. **Amplitude Modulation Detection**
   Amplitude modulation involves varying the strength (amplitude) of an audio signal at a specific frequency, potentially conveying information below the threshold of conscious perception.
   - Identifies patterns in amplitude changes.
   - Reports the dominant modulation frequency and depth.

4. **Stereo Phase Manipulation Detection**
   This involves altering the phase relationship between left and right audio channels to create spatial effects or embed information.
   - Examines phase differences between stereo channels.
   - Reports mean phase difference and correlation between channels.

5. **Binaural Beats Detection**
   Binaural beats are created by playing slightly different frequencies in each ear, which the brain perceives as a beat at the frequency difference
   - Looks for slight frequency differences between stereo channels.
   - Reports detected beat frequency and confidence level.

## Ethical Considerations

The use of subliminal techniques in marketing or political advertising is controversial and may be regulated or prohibited in many jurisdictions. In the United States, the FCC has banned the practice of conveying information to the viewer through the transmittal of messages below the threshold of normal awareness by broadcast licensees since the 70s. Ethical communicators should prioritize transparency and avoid manipulative practices that could undermine societal trust.

The effectiveness of these techniques is debated. Users of this code should approach the topic with caution and ethical consideration.

## Relevance

Detecting these techniques is important for identifying potentially manipulative practices, understanding sophisticated audio marketing techniques used in the industry, ensuring marketing materials adhere to laws and regulations, and verifying no unintended subliminal content is present.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Librosa
- Concurrent.futures (part of Python standard library)
- Logging (part of Python standard library)

You can install the required packages using pip:

```
pip install numpy scipy librosa
```

## Usage

1. Import the `SubliminalAudioDetector` class:

```python
from subliminal2 import SubliminalAudioDetector
```

2. Create an instance of the detector:

```python
detector = SubliminalAudioDetector(chunk_duration=10, overlap=1)
```

3. Analyze an audio file:

```python
report = detector.analyze_audio("path/to/your/audio/file.wav")
```

4. View the generated report:

```python
print(report)
```

## Class: SubliminalAudioDetector

### Constructor Parameters

- `chunk_duration` (float, default=10): Duration of each audio chunk in seconds.
- `overlap` (float, default=1): Overlap between chunks in seconds.

### Methods

#### `analyze_audio(audio_file)`

Analyzes the given audio file for subliminal content and generates a comprehensive report.

- Parameters:
  - `audio_file` (str): Path to the audio file to be analyzed.
- Returns:
  - A string containing a formatted report of the analysis results.

## Performance Optimizations

The script uses chunked processing and parallel execution to efficiently handle large audio files:

- Audio is processed in configurable chunks (default 10 seconds).
- Chunks overlap (default 1 second) to ensure continuity in analysis.
- ThreadPoolExecutor is used for parallel processing of chunks.
- Results from all chunks are aggregated to provide a comprehensive analysis.

## Extending the Detector

To add a new detection technique:

1. Implement a new detector method in the `SubliminalAudioDetector` class.
2. Add the new method to the `initialize_detectors` list.
3. Implement an appropriate aggregation method in `aggregate_results`.
4. Add interpretation logic in the `generate_report` method.

## Limitations and Considerations

- Detection accuracy may vary based on audio quality and content.
- Some legitimate audio effects may be flagged as potential subliminal content.
- Processing time depends on file size, chunk duration, and available computational resources.

## Future Improvements

- Implement progress reporting for long-running analyses.
- Add options for saving and loading intermediate results.
- Develop more sophisticated result aggregation methods.
- Implement adaptive chunking based on audio content.

## Contributing

Contributions to improve the Subliminal Audio Detector are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

MIT License

Copyright (c) 2024 Jacqueline Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

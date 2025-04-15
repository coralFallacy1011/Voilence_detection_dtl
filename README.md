# Real-time Violence Detection System

This project implements a real-time violence detection system using deep learning. The system can process video input from various sources (webcam, video files) and detect violent activities in real-time using a pre-trained deep learning model.

## Features

- Real-time video processing with OpenCV
- Deep learning-based violence detection using a custom CNN model
- Support for multiple input sources (webcam, video files)
- Configurable detection thresholds
- Real-time visualization of detection results
- Pre-trained model support for immediate use
- Easy-to-use command-line interface

## Technical Details

The system uses:
- TensorFlow/Keras for the deep learning model
- OpenCV for video processing
- NumPy for numerical operations
- A custom CNN architecture trained on violence detection datasets

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Setup Steps

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Run the main application with default settings:
```bash
python main.py
```

### Advanced Usage
The application supports several command-line arguments:
```bash
python main.py --source 0  # Use webcam (0 is default webcam)
python main.py --source video.mp4  # Process a video file
python main.py --threshold 0.7  # Set custom detection threshold
python main.py --output output.mp4  # Save processed video
```

### Command Line Arguments
- `--source`: Video source (webcam index or video file path)
- `--threshold`: Detection confidence threshold (0.0 to 1.0)
- `--output`: Output file path for processed video
- `--model`: Path to custom model file
- `--help`: Show help message

## Model Files

The project uses pre-trained models for violence detection:
- `violence_detection_model.h5`: Main model file
- `best_model.h5`: Alternative model with different architecture

Due to size limitations, model files (*.h5) are not included in the repository. Please contact the maintainers for access to the model files.

## Project Structure

```
.
├── main.py                 # Main application script
├── realtime_violence_detection.py  # Core detection module
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## Requirements

See `requirements.txt` for a complete list of Python dependencies. Main dependencies include:
- tensorflow>=2.8.0
- opencv-python>=4.5.5
- numpy>=1.21.0
- scikit-learn>=1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

# Real-time Violence Detection System

// ... existing code ...

## License

MIT License

Copyright (c) 2024 [Harshit Saroha]

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

## Contact

For questions or model file access, please contact:

Email: harshitsaroha22@gmail.com

LinkedIn: https://www.linkedin.com/in/harshitsaroha/

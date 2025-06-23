# Automated Student Attendance System with Age and Gender Prediction

## Project Overview
This project implements an AI-powered face recognition system that automates student attendance tracking while also predicting age and gender. The system addresses limitations of traditional attendance methods by providing a contactless, efficient, and fraud-resistant solution.

## Developed By
Dhruv Menghani, Made Mahatti Prayascita Chandra, Muhamad Aldi Apriansyah, Rivan Meinaki **(Group 5)**

*President University, 2025*

## Background & Problem Statement

### Background
In academic environments, attendance tracking is a critical component of student evaluation. Traditional methods like manual signatures and fingerprint scanners present several challenges:

- Time-consuming, especially for large classes
- Prone to errors and fraud (proxy attendance)
- Hygiene concerns with physical contact systems
- Limited reliability and accuracy

Modern AI and computer vision technologies offer promising alternatives through face recognition, which provides:
- Contactless operation
- Improved efficiency
- Enhanced accuracy and fraud resistance
- Additional capabilities like demographic analysis

### Problem Statement
Traditional attendance systems suffer from:
- Inefficiency and time wastage
- Vulnerability to fraud
- Human error
- Technical limitations (e.g., fingerprint scanners failing with wet/dirty fingers)

Our AI-based automatic attendance system addresses these challenges by:
- Automating the attendance process
- Minimizing human error
- Preventing fraud through secure facial recognition
- Providing real-time verification
- Extending functionality with age and gender prediction for demographic insights

## Technologies Used

- **OpenCV**: Face detection, video frame capture, and image manipulation
- **NumPy**: Image data handling and numerical operations
- **MediaPipe**: Advanced face detection and facial landmark tracking
- **Haarcascade**: Pre-trained model for face detection
- **TensorFlow/Keras**: Deep learning models for gender and age prediction
- **FaceNet**: Face embedding generation for recognition
- **Pandas**: Data storage and management
- **Datetime**: Timestamp generation
- **PIL**: Image processing tasks

## System Components

### 1. Face Data Capture (`camera.py`)
- Opens "Capture Face Data" window
- Captures and saves 5 images of the subject
- Creates a dataset for training

### 2. Face Data Processing (`capture.py`)
- Processes captured face images
- Generates face embeddings
- Creates `data.pkl` file storing all face data

### 3. Main Application (`main.py`)
- Opens "Face Verification" window
- Detects faces in real-time
- Recognizes registered individuals
- Predicts age and gender
- Records attendance with timestamps
- Labels unknown faces appropriately

## Evaluation
The system was evaluated primarily on the gender detection model, which uses a CNN architecture. Various experiments were conducted by:
- Adjusting layer configurations
- Modifying parameters
- Adding dropout layers

The final model was trained with 20 epochs, achieving optimal performance.

## Results
The system successfully:
- Captures face data for registration
- Recognizes registered individuals
- Detects gender (male/female)
- Predicts age with reasonable accuracy
- Records attendance with timestamps
- Identifies unknown individuals

## Conclusion
This Automated Student Attendance System demonstrates how AI and face recognition can transform attendance tracking in educational institutions. The system:

- Reduces fraud and human error
- Saves time compared to manual methods
- Provides additional demographic insights through age and gender prediction
- Creates a more secure and efficient attendance management process

While challenges remain regarding lighting conditions and camera angles, the system shows significant promise for practical implementation. Future improvements could include more advanced AI models for enhanced accuracy and additional features.

## Installation and Usage

### Prerequisites
- Python 3.7+
- Required libraries: OpenCV, NumPy, MediaPipe, TensorFlow, Keras, Pandas, PIL

### Setup
1. Clone the repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the System
1. Capture face data:
   ```
   python camera1.py
   ```
2. Process face data:
   ```
   python capture1.py
   ```
3. Run the main application:
   ```
   python main3.py
   ```

## License
[Specify your license information here]

# Face Emotion Recognition App

This repository contains the source code for the **Face Emotion Recognition** web application. The application is built using **Streamlit** and employs a deep learning model to recognize emotions from facial images.

## Features

- **Emotion Detection from Images**: Upload an image and detect the emotion expressed on the face.
- **Camera Input**: Capture an image using your device's camera for emotion recognition.
- **Real-Time Predictions**: Provides predictions along with confidence scores.
- **Interactive Interface**: Simple and user-friendly navigation between features.

## Recognized Emotions

The model predicts one of the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-emotion-recognition.git
   cd face-emotion-recognition
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the pre-trained model (`5.keras`) in the root directory.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the **Emotion Detection** tab:
   - Upload a facial image using the file uploader.
2. Alternatively, navigate to the **Camera Input** tab:
   - Use your device's camera to capture a facial image.
3. Wait for the model to process the image and display the predicted emotion and confidence level.

## File Structure

```
face-emotion-recognition/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── 5.keras                 # Pre-trained model (not included in repo)
├── README.md               # Project documentation
```

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- PIL (Pillow)
- NumPy

For a full list of dependencies, see `requirements.txt`.

## Model

The application uses a pre-trained deep learning model saved as `5.keras`. The model was trained on the FER-2013 dataset and outputs probabilities for each emotion class.

## Contribution

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The FER-2013 dataset for training the emotion recognition model.
- The developers of Streamlit for providing an easy-to-use framework for building web apps.

## Author

**Rewan Abdulkariem** ❤️

Feel free to connect for questions or collaborations!

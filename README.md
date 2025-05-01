
# **_Violence and Non-Violence Detection System_**

This project is a deep learning-based system to detect **violence** or **non-violence** in video streams (real-time webcam or pre-recorded video) using **TensorFlow/Keras** and **OpenCV**.

---

## **_Project Structure_**

- `train_model.py` — Script to extract frames, train the CNN model, and save it  
- `predict_video.py` — Real-time/video file detection using the trained model  
- `README.md` — Project documentation  

---

## **_🔧 Installation_**

Install all required dependencies using pip:

```bash
pip install tensorflow tensorflow-gpu numpy matplotlib pillow opencv-python scikit-learn
```

---

## **_Model Training (`train_model.py`)_**

This script:

- Extracts frames from labeled videos (Violence/NonViolence)
- Preprocesses and augments the data
- Builds a Convolutional Neural Network (CNN)
- Trains the model and saves it as `violence_detection_model002.h5`

Ensure the dataset paths are configured correctly in the script.

---

## **_Inference with Video (`predict_video.py`)_**

This script:

- Loads the trained model
- Opens webcam or video file (`violence_1.mp4` by default)
- Predicts "Violence" or "Non-Violence" on each frame
- Displays prediction labels and confidence in real time
- Shows Frames Per Second (FPS)

To run:

```bash
python predict_video.py
```

Press `q` to exit the live stream window.

---

## **_Dataset Used_**

**Real Life Violence Situations Dataset**  
Download here: [https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

- Contains two classes: `Violence` and `NonViolence`
- Videos should be extracted into individual frames for training and validation

---

## **_Model Architecture_**

- **Input Shape:** 224x224x3
- **Architecture:**
  - Conv2D → ReLU → MaxPooling
  - Conv2D → ReLU → MaxPooling
  - Conv2D → ReLU → MaxPooling
  - Flatten → Dense(128, ReLU) → Dense(1, Sigmoid)
- **Output:** Binary class — `1` for Violence, `0` for Non-Violence

---

## **_Evaluation_**

- Accuracy, precision, recall, F1 score using `sklearn`
- You can modify the training script to save confusion matrix and reports

---

## **_Running the Project_**

1. Clone the repo:

    ```bash
    git clone https://github.com/keshavag4rwal/violenceNonViolenceDetectionSystem.git
    cd violenceNonViolenceDetectionSystem
    ```

2. Ensure the trained model (`violence_detection_model002.h5`) is in the root directory.

3. Run the detection script:

    ```bash
    python predict_video.py
    ```

---

## **_Future Improvements_**

- Add alert system (buzzer/email notification) on detection
- Host the model on a web interface (Flask or Streamlit)
- Fine-tune a pretrained model (e.g., MobileNet, ResNet) for better accuracy

---

## **_Contact_**

For any inquiries or collaboration:

**Name**  
 [keshavagarwal2004@gmail.com]  
🔗 [https://github.com/keshavag4rwal]

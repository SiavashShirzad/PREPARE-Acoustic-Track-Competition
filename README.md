# Project Title

This repository provides a pipeline for processing audio files, extracting features, training a machine learning model, and running a Gradio-based application for inference for PREPARE Acoustic Track Competition regarding audio classification and detection of dementia.

---

## Repository Structure

```plaintext
Demo/
  Demo.py  # Gradio application for running inference
Features/  # Directory where extracted features are stored
Model/
  model.joblib  # Pre-trained model file
train_audios/  # Directory for raw training audio files
test_audios/   # Directory for raw test audio files
FeatureExtraction.ipynb  # Script for extracting features from audio files
training.ipynb  # Script for training the model
validation.ipynb  # Script for validating the model
requirements.txt  # List of required Python libraries
```

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SiavashShirzad/PREPARE-Acoustic-Track-Competition
   cd PREPARE-Acoustic-Track-Competition
   ```

2. **Install Dependencies**
   Ensure you have Python installed on your system. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Extract Data**
   - Download the raw audio data and extract the files into the corresponding directories:
     - Training audio files: Place them in the `train_audios/` directory.
     - Test audio files: Place them in the `test_audios/` directory.
   - Ensure any other necessary data files (e.g., validation labels) are placed in the root directory.

---

## Workflow

### 1. **Feature Extraction**
   Extract features from both training and testing audio files using the `FeatureExtraction.ipynb` notebook.
   
   - Open the notebook:
     ```bash
     jupyter notebook FeatureExtraction.ipynb
     ```
   - Follow the steps in the notebook to process the audio files and save the extracted features in the `Features/` directory.

### 2. **Training**
   Train the model using the `training.ipynb` notebook.
   
   - Open the notebook:
     ```bash
     jupyter notebook training.ipynb
     ```
   - Follow the steps to train the model and save the trained model file in the `Model/` directory.

### 3. **Validation**
   Validate the model using the `validation.ipynb` notebook.
   
   - Ensure you have validation labels available in the root directory.
   - Open the notebook:
     ```bash
     jupyter notebook validation.ipynb
     ```
   - Follow the steps in the notebook to evaluate the model on the validation dataset.

### 4. **Running the Gradio Application**
   Use the `Demo/Demo.py` script to run a browser-based interface for inference.

   - Ensure the trained model (`model.joblib`) is available in the `Model/` directory.
   - Run the Gradio application:
     ```bash
     python Demo/Demo.py
     ```
   - This will start a local server and open a Gradio interface in your browser, where you can test the model with audio inputs.

---

## Additional Notes

- Ensure the directory structure is maintained as described for the scripts to function correctly.
- If you encounter any issues or have questions, please refer to the documentation in the respective notebooks or scripts.

---

## Contact

For any issues or further questions, feel free to open an issue in the repository or contact me using the email below.

Siavash Shirzadeh Barough M.D.
sshirza1@jhmi.edu

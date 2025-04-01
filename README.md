# Machine Learning Pipeline for Classifying Audio Files into Three Dementia Stages

This repository presents a comprehensive pipeline for the classification of audio recordings into three stages of dementia. The framework includes audio preprocessing, feature extraction, machine learning model training, and deployment via a Gradio-based application. Developed for the PREPARE Acoustic Track Competition, this approach aims to advance audio-based dementia detection and classification methodologies.

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
inference.ipynb # notebook to run the model and the analysis on data other than training and validation datasets.
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

   Ensure you have Python and ipynb-kernel(jupyter) installed on your system. Install the required Python libraries by running in the environment:
   ```bash
   pip install -r requirements.txt
   ```
   or
   for each notebook there is a cell (first cell) containing this code which will install the dependencies required for the code:
   ```bash
   pip install -r requirements.txt
   ```
   If using conda install jupyter using:
   ```bash
   conda install ipykernel
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
   Training and Testing features should be extracted seperatly by setting the dataType variable in this notebook to train and test and running the notebook for each.
   
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

### **Inferencing new audio files**
   Use the `inference.ipynb` notebook you can run the model on a folder containing a set of audio files.

   - Ensure the trained model (`model.joblib`) is available in the `Model/` directory.
   - Ensure ipykernel is installed
   - Ensure the metadata file is available and all the audio files are in a folder.
   - change the adress variables inside the notebook to the desired directories and run thm all.
   - the results will be saved in inference_results.csv file.
   - Feature extraction is included in this code and there is no need to run it.
---

## Additional Notes

- Ensure the directory structure is maintained as described for the scripts to function correctly.
- If you encounter any issues or have questions, please refer to the documentation in the respective notebooks or scripts.

---

## Contact

For any issues or further questions, feel free to open an issue in the repository or contact me using the email below.

Siavash Shirzadeh Barough M.D.
sshirza1@jhmi.edu

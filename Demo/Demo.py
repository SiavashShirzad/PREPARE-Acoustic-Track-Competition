import os
import torchaudio
import pandas as pd
import torch
from tqdm import tqdm
import opensmile
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModel
from transformers import WhisperFeatureExtractor, WhisperModel
import gradio as gr
import joblib
import tempfile
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
from PIL import Image  # Import PIL Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained model
model = joblib.load('../Model/model.joblib')
print("Model Loaded")

# Define feature extraction functions

def load_and_resample_audio(file_path, target_sampling_rate):
    waveform, original_rate = torchaudio.load(file_path)
    # Compute original length in seconds
    original_length = waveform.shape[-1] / original_rate

    if original_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_sampling_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(), original_length

def audio_embeddings_model(model_name):
    if model_name == "compare":
        model = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    elif model_name == "egemaps":
        model = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return model

def audio_embeddings(audio_list, model, sampling_rate):
    embeddings_list = []
    for audio in tqdm(audio_list, desc="Extracting OpenSMILE features"):
        embeddings = model.process_signal(audio.numpy(), sampling_rate)
        embeddings_list.append(embeddings.values.flatten())
    return embeddings_list

def extract_embeddings_from_transformer_model(audio_list, model_name, sampling_rate):
    print(f"Extracting embeddings using {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    transformer_model = AutoModel.from_pretrained(model_name).to(device)
    transformer_model.eval()

    embeddings_list = []
    for audio in tqdm(audio_list, desc=f"Extracting {model_name} embeddings"):
        inputs = feature_extractor(audio.numpy(), sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            # Mean pooling over time
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings.squeeze())
    return embeddings_list

def extract_whisper_embeddings(audio_list, model_name, sampling_rate):
    print(f"Extracting embeddings using {model_name} (Whisper) ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    whisper_model = WhisperModel.from_pretrained(model_name).to(device)
    whisper_model.eval()

    embeddings_list = []
    for audio in tqdm(audio_list, desc=f"Extracting {model_name} embeddings"):
        # Convert audio to numpy, in case it's a tensor
        audio_np = audio.numpy()
        inputs = feature_extractor(audio_np, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = whisper_model.encoder(**inputs)
            # Mean pooling over time
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings.squeeze())
    return embeddings_list

def extract_additional_features(audio, sampling_rate):
    # Convert the audio tensor to a numpy array
    audio = audio.numpy()

    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)

    # Extract Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)

    # Extract Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate, roll_percent=0.85)

    # Extract Zero Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)

    # Extract Pitch (using librosa's piptrack for pitch tracking)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sampling_rate)
    pitch = np.max(pitches, axis=0)

    # Return the features as a flat vector
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_rolloff, axis=1),
        np.mean(zero_crossings, axis=1),
        np.mean(pitch)
    ])
    return features

def create_pie_chart(probabilities):
    labels = ['Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
    colors = ['#4CAF50', '#FFEB3B', '#F44336']  # Green, Yellow, Red

    # Handle case with no data
    if probabilities is None or sum(probabilities) == 0:
        probabilities = [1, 1, 1]  # Equal probabilities
        labels = ['No Data', 'No Data', 'No Data']
        colors = ['#CCCCCC', '#CCCCCC', '#CCCCCC']

    fig, ax = plt.subplots(figsize=(6,6))
    wedges, texts, autotexts = ax.pie(
        probabilities, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        textprops=dict(color="w")
    )

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')  

    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Convert BytesIO buffer to PIL Image
    img = Image.open(buf).convert("RGB")  # Ensure it's in RGB mode
    return img

def create_bar_chart(probabilities):
    labels = ['Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
    colors = ['#4CAF50', '#FFEB3B', '#F44336']  # Green, Yellow, Red

    # Handle case with no data
    if probabilities is None or sum(probabilities) == 0:
        probabilities = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0, 1)  # Since probabilities range from 0 to 1
    ax.set_ylabel('Probability')
    ax.set_title('Average Class Probabilities')

    # Add probability labels on top of each bar
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.annotate(f'{prob*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontweight='bold')

    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Convert BytesIO buffer to PIL Image
    img = Image.open(buf).convert("RGB")  # Ensure it's in RGB mode
    return img

# Main processing function
def process_files(uploaded_files, gender, age):
    # Check if any files are uploaded
    if not uploaded_files:
        return None, "No files uploaded."

    original_sampling_rate = 48000  # Original sampling rate of MP3 files
    target_sampling_rate = 16000    # Target sampling rate for models

    audio_list = []
    file_names = []

    # Process each uploaded file
    for file_path in uploaded_files:
        # Validate file type
        if not file_path.lower().endswith('.mp3'):
            return None, f"Invalid file type: {file_path}. Please upload MP3 files only."
        
        waveform, length = load_and_resample_audio(file_path, target_sampling_rate)
        audio_list.append(waveform)
        file_names.append(os.path.basename(file_path))

    # Extract features from original audio using eGeMAPS
    print("Loading eGeMAPS model...")
    egemaps_model = audio_embeddings_model("egemaps")

    print("Extracting features from original audio (eGeMAPS)...")
    egemaps_features = audio_embeddings(audio_list, egemaps_model, target_sampling_rate)

    # Transformer-based models
    wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self"
    whisper_model_name = "openai/whisper-medium"

    print("Extracting Wav2Vec2 embeddings...")
    wav2vec2_features = extract_embeddings_from_transformer_model(audio_list, wav2vec2_model_name, target_sampling_rate)

    print("Extracting Whisper encoder embeddings...")
    whisper_features = extract_whisper_embeddings(audio_list, whisper_model_name, target_sampling_rate)

    # Extract additional features (MFCC, Chroma, Spectral Centroid, etc.)
    additional_features_list = []
    for audio in tqdm(audio_list, desc="Extracting additional features"):
        additional_features = extract_additional_features(audio, target_sampling_rate)
        additional_features_list.append(additional_features)

    # Create a DataFrame of additional features
    additional_feature_columns = (
        [f"mfcc_{i}" for i in range(1, 14)]
        + [f"chroma_{i}" for i in range(1, 13)]
        + ["spectral_centroid", "spectral_rolloff", "zero_crossing_rate", "pitch"]
    )
    additional_features_df = pd.DataFrame(additional_features_list, columns=additional_feature_columns)

    # Create DataFrame for all features
    features_df = pd.DataFrame(egemaps_features, columns=egemaps_model.feature_names)
    features_df["Wav2Vec2_embeddings"] = list(wav2vec2_features)
    features_df["Whisper_embeddings"] = list(whisper_features)

    # Concatenate additional features onto the same DataFrame
    features_df = pd.concat([features_df, additional_features_df], axis=1)

    # Expand Wav2Vec2 embeddings
    wav2vec2_length = len(features_df['Wav2Vec2_embeddings'][0])
    wav2vec2_df = pd.DataFrame(features_df['Wav2Vec2_embeddings'].tolist(), 
                               columns=[f'Embedding1_{i+1}' for i in range(wav2vec2_length)])
    features_df = pd.concat([features_df.drop(columns=['Wav2Vec2_embeddings']), wav2vec2_df], axis=1)

    # Expand Whisper embeddings
    whisper_length = len(features_df['Whisper_embeddings'][0])
    whisper_df = pd.DataFrame(features_df['Whisper_embeddings'].tolist(), 
                               columns=[f'Embedding2_{i+1}' for i in range(whisper_length)])
    features_df = pd.concat([features_df.drop(columns=['Whisper_embeddings']), whisper_df], axis=1)

    # Encode gender and add age
    features_df["age"] = age
    gender_encoded = 1 if gender.lower() == "male" else 0
    features_df["gender"] = gender_encoded

    # Optionally, apply scaling or other preprocessing here
    # For example:
    # scaler = StandardScaler()
    # features_df = scaler.fit_transform(features_df)

    # Make predictions with probabilities
    try:
        probabilities = model.predict_proba(features_df)  # Shape: (n_samples, n_classes)
    except AttributeError:
        return None, "The loaded model does not support predict_proba."

    # Create probability columns
    prob_df = pd.DataFrame(probabilities, columns=['prob_normal', 'prob_mci', 'prob_ad'])
    features_df = pd.concat([features_df, prob_df], axis=1)

    # Prepare results by combining features with probabilities
    results = features_df.copy()
    results["file_name"] = file_names

    # Aggregate probabilities across all files (e.g., average)
    avg_probabilities = probabilities.mean(axis=0)  # Shape: (n_classes,)

    # Create pie chart based on average probabilities
    pie_chart = create_pie_chart(avg_probabilities)

    # Create bar chart based on average probabilities (optional)
    bar_chart = create_bar_chart(avg_probabilities)

    # Create a summary string with average probabilities
    summary = (
        f"**Average Class Probabilities:**\n\n"
        f"- **Normal (0):** {avg_probabilities[0]*100:.2f}%\n"
        f"- **Mild Cognitive Impairment (1):** {avg_probabilities[1]*100:.2f}%\n"
        f"- **Alzheimer's Disease (2):** {avg_probabilities[2]*100:.2f}%"
    )

    # Optionally, prepare a CSV with detailed probabilities per file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        results.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    return pie_chart, bar_chart, summary, csv_path

# Define Gradio interface
def main():
    title = "MP3 Feature Extraction and Classification App"
    description = (
        "Drag and drop your MP3 files below. The app will extract various audio features, take additional inputs "
        "(gender and age), and provide a visual summary of classification probabilities."
    )

    iface = gr.Interface(
        fn=process_files,
        inputs=[
            gr.File(
                label="Upload MP3 Files",
                file_count="multiple",
                type="filepath"  # Accepts file paths
            ),
            gr.Radio(
                label="Select Gender",
                choices=["Male", "Female"],
                value="Male",  # Default value
                type="value"
            ),
            gr.Number(
                label="Enter Age",
                value=50,  # Default age
                precision=0
            )
        ],
        outputs=[
            gr.Image(label="Classification Results (Pie Chart)"),
            gr.Image(label="Classification Results (Bar Chart)"),
            gr.Markdown(label="Prediction Summary"),
            gr.File(label="Download Detailed Probabilities CSV")
        ],
        title=title,
        description=description,
        allow_flagging="never",
        examples=None,
        # Optional: you can set theme or other UI parameters here
    )

    iface.launch()

if __name__ == "__main__":
    main()

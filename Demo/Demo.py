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
from PIL import Image  
import shap
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = joblib.load('../Model/model.joblib')
print("Model Loaded")


def load_and_resample_audio(file_path, target_sampling_rate):
    waveform, original_rate = torchaudio.load(file_path)
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
        audio_np = audio.numpy()
        inputs = feature_extractor(audio_np, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = whisper_model.encoder(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings.squeeze())
    return embeddings_list

def extract_additional_features(audio, sampling_rate):
    audio = audio.numpy()

    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate, roll_percent=0.85)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sampling_rate)
    pitch = np.max(pitches, axis=0)
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_rolloff, axis=1),
        np.mean(zero_crossings, axis=1),
        np.mean(pitch)
    ])
    return features

def create_bar_chart(probabilities):
    labels = ['Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
    colors = ['#4CAF50', '#FFEB3B', '#F44336']  

    if probabilities is None or sum(probabilities) == 0:
        probabilities = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Average Class Probabilities')

    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.annotate(f'{prob*100:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")  
    return img


def categorize_feature(feature_name):
    """
    Categorize each feature into broad groups:
    - Wav2Vec
    - Whisper
    - OpenSMILE
    - Age
    - Gender
    """
    if feature_name.startswith('Embedding1_'):
        return 'Wav2Vec'
    elif feature_name.startswith('Embedding2_'):
        return 'Whisper'
    elif 'age' in feature_name.lower():
        return 'Age'
    elif 'gender' in feature_name.lower():
        return 'Gender'
    else:
        return 'OpenSMILE'

def generate_shap_charts_for_first_sample(features_df, model):
    """
    Generates 3 SHAP bar charts (one per class) for the first sample in features_df.
    Returns a list of PIL Images: [img_class0, img_class1, img_class2].
    If something fails (e.g. no 'select' or 'xgb' steps), returns None.
    """
    try:
        selector = model.named_steps['select']     
        xgb_model_final = model.named_steps['xgb']  
        print("Model does not have named steps 'select' or 'xgb'. SHAP explanation cannot be generated.")
        return None
    
    original_feature_names = features_df.columns.tolist()
    X_selected = selector.transform(features_df)
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [original_feature_names[i] for i in selected_indices]
    
    explainer = shap.TreeExplainer(xgb_model_final)
    shap_values_full = explainer.shap_values(X_selected)  

    if len(shap_values_full.shape) != 3:
        print("SHAP values do not have the expected shape for a multi-class model.")
        return None
    
    num_samples = shap_values_full.shape[0]
    num_classes = shap_values_full.shape[2]
    
    if num_samples < 1:
        print("No samples to explain in features_df.")
        return None

    sample_index = 0

    shap_values_for_sample = shap_values_full[sample_index, :, :]  
    
    shap_images = []

    feature_categories = [categorize_feature(fn) for fn in selected_feature_names]
    unique_categories = sorted(list(set(feature_categories)))
    
    category_palette = {
        'Wav2Vec': '#1f77b4', 
        'Whisper': '#ff7f0e',  
        'OpenSMILE': '#2ca02c',
        'Age': '#9467bd',       
        'Gender': '#8c564b'     
    }
    for cat in unique_categories:
        if cat not in category_palette:
            category_palette[cat] = '#7f7f7f'
    
    for class_index in range(num_classes):
        shap_1d = shap_values_for_sample[:, class_index]  
        
        category_shap_sums = {cat: 0.0 for cat in unique_categories}
        for feat_idx, cat in enumerate(feature_categories):
            category_shap_sums[cat] += shap_1d[feat_idx]
        
        df_category_shap = pd.DataFrame({
            'Category': list(category_shap_sums.keys()),
            'SHAP': list(category_shap_sums.values())
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        df_category_shap = df_category_shap.reindex(
            df_category_shap['SHAP'].abs().sort_values(ascending=False).index
        )

        sns.barplot(
            data=df_category_shap,
            x='SHAP',
            y='Category',
            palette=[category_palette[cat] for cat in df_category_shap['Category']]
        )
        ax.axvline(0, color='grey', linewidth=0.8)
        ax.set_title(f"SHAP for Class {class_index}", fontsize=14)
        ax.set_xlabel("SHAP Contribution", fontsize=12)
        ax.set_ylabel("Feature Category", fontsize=12)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        shap_images.append(img)
    
    return shap_images


# Main processing function
def process_files(uploaded_files, recorded_audio, gender, age):
    all_files = []
    file_names = []

    if uploaded_files:
        for file_path in uploaded_files:
            if not file_path.lower().endswith(('.mp3', '.wav')):
                return None, "Invalid file type uploaded. Please upload MP3 or WAV files only."
            all_files.append(file_path)
            file_names.append(os.path.basename(file_path))

    if recorded_audio:
        if not recorded_audio.lower().endswith('.wav'):
            return None, "Invalid recorded file type. Please record audio in WAV format."
        all_files.append(recorded_audio)
        file_names.append(os.path.basename(recorded_audio))

    if not all_files:
        return None, "No audio files provided. Please upload MP3/WAV files or record an audio."

    target_sampling_rate = 16000 

    audio_list = []

    # Process each audio file
    for file_path in all_files:
        waveform, length = load_and_resample_audio(file_path, target_sampling_rate)
        max_length = target_sampling_rate * 30 
        if waveform.shape[0] > max_length:
            waveform = waveform[:max_length]
        elif waveform.shape[0] < max_length:
            padding = max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        audio_list.append(waveform)

    print("Loading eGeMAPS model...")
    egemaps_model = audio_embeddings_model("egemaps")

    print("Extracting features from original audio (eGeMAPS)...")
    egemaps_features = audio_embeddings(audio_list, egemaps_model, target_sampling_rate)

    wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self"
    whisper_model_name = "openai/whisper-medium"

    print("Extracting Wav2Vec2 embeddings...")
    wav2vec2_features = extract_embeddings_from_transformer_model(audio_list, wav2vec2_model_name, target_sampling_rate)

    print("Extracting Whisper encoder embeddings...")
    whisper_features = extract_whisper_embeddings(audio_list, whisper_model_name, target_sampling_rate)

    additional_features_list = []
    for audio in tqdm(audio_list, desc="Extracting additional features"):
        additional_features = extract_additional_features(audio, target_sampling_rate)
        additional_features_list.append(additional_features)

    additional_feature_columns = (
        [f"mfcc_{i}" for i in range(1, 14)]
        + [f"chroma_{i}" for i in range(1, 13)]
        + ["spectral_centroid", "spectral_rolloff", "zero_crossing_rate", "pitch"]
    )
    additional_features_df = pd.DataFrame(additional_features_list, columns=additional_feature_columns)

    features_df = pd.DataFrame(egemaps_features, columns=egemaps_model.feature_names)
    features_df["Wav2Vec2_embeddings"] = list(wav2vec2_features)
    features_df["Whisper_embeddings"] = list(whisper_features)

    features_df = pd.concat([features_df, additional_features_df], axis=1)

    wav2vec2_length = len(features_df['Wav2Vec2_embeddings'][0])
    wav2vec2_df = pd.DataFrame(
        features_df['Wav2Vec2_embeddings'].tolist(),
        columns=[f'Embedding1_{i+1}' for i in range(wav2vec2_length)]
    )
    features_df = pd.concat([features_df.drop(columns=['Wav2Vec2_embeddings']), wav2vec2_df], axis=1)

    whisper_length = len(features_df['Whisper_embeddings'][0])
    whisper_df = pd.DataFrame(
        features_df['Whisper_embeddings'].tolist(),
        columns=[f'Embedding2_{i+1}' for i in range(whisper_length)]
    )
    features_df = pd.concat([features_df.drop(columns=['Whisper_embeddings']), whisper_df], axis=1)

    features_df["age"] = age
    gender_encoded = 1 if gender.lower() == "male" else 0
    features_df["gender"] = gender_encoded

    try:
        probabilities = model.predict_proba(features_df)  
    except AttributeError:
        return None, "The loaded model does not support predict_proba."

    results = features_df.copy()
    results["file_name"] = file_names
    avg_probabilities = probabilities.mean(axis=0)  

    bar_chart = create_bar_chart(avg_probabilities)

    summary = (
        f"**Average Class Probabilities:**\n\n"
        f"- **Normal (0):** {avg_probabilities[0]*100:.2f}%\n"
        f"- **Mild Cognitive Impairment (1):** {avg_probabilities[1]*100:.2f}%\n"
        f"- **Alzheimer's Disease (2):** {avg_probabilities[2]*100:.2f}%"
    )

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        results.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    shap_images = generate_shap_charts_for_first_sample(features_df.iloc[[0]], model)
    if shap_images is None:
        return bar_chart, summary, csv_path, None, None, None

    return bar_chart, summary, csv_path, shap_images[0], shap_images[1], shap_images[2]

# Define Gradio interface
def main():
    title = "MP3 Feature Extraction and Classification App"
    description = (
        "Upload your MP3/WAV files or record an audio clip below. The app will extract various audio features, "
        "take additional inputs (gender and age), and provide a visual summary of classification probabilities.\n\n"
        "Additionally, we generate SHAP explanation charts (by feature category) for the first audio sample."
    )

    iface = gr.Interface(
        fn=process_files,
        inputs=[
            gr.File(
                label="Upload MP3/WAV Files",
                file_count="multiple",
                type="filepath"  
            ),
            gr.Audio(
                sources="microphone",
                label="Or Record an Audio Clip (Max 30 seconds)",
                type="filepath"  
            ),
            gr.Radio(
                label="Select Gender",
                choices=["Male", "Female"],
                value="Male",  
                type="value"
            ),
            gr.Number(
                label="Enter Age",
                value=50,  
                precision=0
            )
        ],

        outputs=[
            gr.Image(label="Classification Results (Bar Chart)"),
            gr.Markdown(label="Prediction Summary"),
            gr.File(label="Download Detailed Probabilities CSV"),
            gr.Image(label="SHAP Explanation for Class 0"),
            gr.Image(label="SHAP Explanation for Class 1"),
            gr.Image(label="SHAP Explanation for Class 2")
        ],
        title=title,
        description=description,
        allow_flagging="never"
    )

    iface.launch()

if __name__ == "__main__":
    main()

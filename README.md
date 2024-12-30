# Animal Detection Project

## Overview 🐾🎨🔗
This project aims to detect and classify different animal species using deep learning. Leveraging the ResNet50 architecture, the model is fine-tuned on a dataset of animal images to achieve high accuracy in classification. A Streamlit web application is also included to allow users to upload images and receive predictions in real-time.

## Features 🚀🌟🔧
- **Deep Learning Model**: Utilizes a pre-trained ResNet50 model fine-tuned for animal classification.
- **Data Augmentation**: Enhances the dataset with various transformations for robust training.
- **Streamlit Web App**: Provides an intuitive interface for users to interact with the model.
- **GPU Support**: Optimized for GPU training to speed up computations.

## Project Structure 📁🌍🔗
- `main.py`: Contains the training pipeline for the model.
- `streamlit_app.py`: Hosts the Streamlit web application for real-time predictions.
- `DL_project.keras`: The trained model file.
- `class_names.npy`: File containing the class labels.

## Getting Started 🔄🛠️💲

### Prerequisites 📊📝🔧
Ensure you have the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- PIL (Pillow)
- Streamlit
- patool
- shutil

You can install these dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib pillow streamlit patool
```

### Dataset Preparation 🔐📚🌐
1. Place your dataset in a `.rar` file with the folder structure:
  
2. Update the path to your `.rar` file in the `train_model` function of `main.py`.

### Training the Model 🎓🔢🔄
Run the `main.py` script to train the model:

```bash
python main.py
```

### Streamlit Web Application 📲🌐🌄
To run the web application:

1. Place the trained model (`DL_project.keras`) and class labels file (`class_names.npy`) in the project directory.
2. Update the `MODEL_PATH` and `CLASS_NAMES_PATH` variables in `streamlit_app.py` to reflect the correct paths.
3. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

4. Access the app in your browser at `http://localhost:8501/`.

## Usage 🌍🔦🔧
- **Training**: Customize the `train_model` function in `main.py` to adapt to different datasets.
- **Web App**: Upload an image through the Streamlit interface to classify it and view the predicted class with a confidence score.

## Team Members 👤🌈🔗
- Ahmed Soudy Tawfik Ahmed
- Mustafa Gaser Mekhemar
- Elsayed Osama Elsayed
- Mahmoud Foad Sleem
- Islam Ragab Ahmed
- Ahmed Reda Farag

## Results 🏆🎨🔠
- **Model Accuracy**: Achieved a high validation accuracy through fine-tuning.
- **Loss Metrics**: Demonstrated convergence with minimal overfitting as visualized in the training history plots.

## Future Work 📊🌐🔧
- Expanding the dataset for broader classification.
- Integrating additional architectures for comparative analysis.
- Deploying the app on cloud platforms for wider accessibility.



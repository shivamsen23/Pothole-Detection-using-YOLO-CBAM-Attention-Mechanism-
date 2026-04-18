# Pothole-Detection-using-YOLO+CBAM-Attention-Mechanism


# 📌 Overview
This project presents a deep learning-based pothole detection system using **YOLOv8** enhanced with **CBAM (Convolutional Block Attention Module)**. The system detects potholes in road images and visualizes results through an interactive **Streamlit web application**.

The project demonstrates:
- Object detection using YOLOv8
- Attention mechanism (CBAM) integration
- Visualization via bounding boxes and heatmaps
- Deployment using Streamlit

---

# 🚀 Features
-  Real-time pothole detection  
- CBAM-enhanced feature learning  
-  Evaluation metrics (Precision, Recall, mAP)  
-  Streamlit-based user interface  
-  Heatmap visualization (model attention)  
-  Google Colab support for model loading  

---

## 📁 Project Structure
project-root/
│
├── app.py # Streamlit main application
├── predictor.py # Inference logic (YOLO detection)
├── model_loader.py # Model loading utilities
├── utils.py # Helper functions
├── pothole_data.yaml # Dataset configuration
├── requirements.txt # Dependencies
├── runs/ # YOLO outputs (training/validation)
├── pycache/ # Python cache files



---

## ⚙️ Setup Instructions

# Step 1: Extract Project
Download and extract the project folder to your local system.

# Step 2: Create Virtual Environment (Anaconda)

conda create -n pothole_env python=3.10 -y
conda activate pothole_env

# Step 3: Install Dependencies
pip install -r requirements.txt

# Step 4: Run Streamlit Application
streamlit run app.py

# If the model does not load properly, follow the steps below:

1. Use Google Colab to Load Model
2. Open the provided .ipynb notebook file
3. Upload it to Google Colab
4. Run all cells in the notebook
5. The trained model (best.pt) will be downloaded or generated
6. Save the model to your Google Drive
7. Update Model Path

After saving the model, update the path in your code:
model_path = "/content/drive/MyDrive/pothole_model/best.pt"

# Restart Application
streamlit run app.py

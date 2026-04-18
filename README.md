
# 🛣️ Pothole Detection using YOLOv8 + CBAM Attention Mechanism

---

## 📌 Overview
This project presents a deep learning-based pothole detection system using **YOLOv8** enhanced with **CBAM (Convolutional Block Attention Module)**. The system detects potholes in road images and visualizes results through an interactive **Streamlit web application**.

### Key Highlights:
- Object detection using YOLOv8  
- Integration of CBAM attention mechanism  
- Visualization using bounding boxes and heatmaps  
- Deployment via Streamlit  

---

## 🚀 Features
-  Real-time pothole detection  
-  CBAM-enhanced feature learning  
-  Evaluation metrics (Precision, Recall, mAP)  
-  Streamlit-based user interface  
-  Heatmap visualization (model attention)  
-  Google Colab support for model loading  

---

## 📁 Project Structure

```

project-root/
│
├── app.py                # Streamlit main application
├── predictor.py          # Inference logic (YOLO detection)
├── model_loader.py       # Model loading utilities
├── utils.py              # Helper functions
├── pothole_data.yaml     # Dataset configuration
├── requirements.txt      # Dependencies
├── runs/                 # YOLO outputs (training/validation)
├── **pycache**/          # Python cache files

````

---

## ⚙️ Setup Instructions

### 🧱 Step 1: Extract Project
Download and extract the project folder to your local system.

---

### 🐍 Step 2: Create Virtual Environment (Anaconda)

```bash
conda create -n pothole_env python=3.10 -y
conda activate pothole_env
````

---

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ Step 4: Run Streamlit Application

```bash
streamlit run app.py
```

---

##  Model Loading Issue (If Occurs)

If the model does not load properly, follow these steps:

###  Use Google Colab

1. Open the provided `.ipynb` notebook file
2. Upload it to **Google Colab**
3. Run all cells
4. The trained model (`best.pt`) will be generated/downloaded
5. Save the model to your **Google Drive**

---

###  Update Model Path

Update the model path in your code:

```python
model_path = "/content/drive/MyDrive/pothole_model/best.pt"
```

---

###  Restart Application

```bash
streamlit run app.py
```

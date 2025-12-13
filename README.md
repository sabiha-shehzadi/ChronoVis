# ChronoVis – Real-Time Age & Gender Prediction

![ChronoVis](https://img.shields.io/badge/Status-Completed-green)

**ChronoVis** is a real-time age and gender prediction application using deep learning and computer vision. Built with TensorFlow, OpenCV, and Gradio, this project demonstrates a complete ML pipeline: from data preprocessing and model training to deployment as an interactive web app.

---

## 🚀 Features

* **Real-Time Prediction:** Predict age and gender using webcam or uploaded images.
* **Face Detection:** Uses Haar Cascade for detecting faces, with fallback to full image if no face is detected.
* **Dual-Purpose Model:** Single CNN predicts both age (regression) and gender (classification).
* **User-Friendly Interface:** Built with Gradio for a simple, interactive UI.
* **Portable Deployment:** Fully deployable on Hugging Face Spaces for instant web access.

---

## 🛠 Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Web Interface:** Gradio
* **Deployment:** Hugging Face Spaces
* **Dataset:** UTKFace (subset of 10,000 images for RAM efficiency)

---

## 📂 Project Structure

```
ChronoVis/
├── dataset/                     # Preprocessed dataset folder
├── Age_Gender_Model.h5          # Trained CNN model
├── app.py                       # Gradio web interface code
├── requirements.txt             # Dependencies for Hugging Face deployment
└── README.md                    # Project documentation
```

---

## ⚙ Installation

1. **Clone this repository**

```bash
git clone https://github.com/SabihaKhan/ChronoVis.git
cd ChronoVis
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app locally**

```bash
python app.py
```

4. **Visit** `http://127.0.0.1:7860` in your browser to interact with the interface.

---

## 🖼 Usage

* Use your **webcam** for real-time predictions.
* Upload any face image to get **age and gender estimates**.
* Confidence scores are provided for gender prediction.

Example Output:

```
✅ Face Detected
Gender: Female (97.5%)
Estimated Age: 28 years
```

---

## 📈 Model Details

* **Architecture:** CNN with 3 convolutional layers, batch normalization, max pooling, and fully connected dense layers.
* **Outputs:**

  * Gender (binary classification, sigmoid activation)
  * Age (regression, ReLU activation)
* **Loss Functions:** Binary cross-entropy for gender, mean absolute error (MAE) for age
* **Training Data:** 10,000 images from the UTKFace dataset
* **Input Size:** 128x128 RGB images

---

## 📊 Model Performance

### **1. Gender Classification**

| Gender           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Male             | 0.89      | 0.84   | 0.86     | 1071    |
| Female           | 0.82      | 0.88   | 0.85     | 929     |
| **Accuracy**     | -         | -      | **0.86** | 2000    |
| **Macro Avg**    | 0.86      | 0.86   | 0.86     | 2000    |
| **Weighted Avg** | 0.86      | 0.86   | 0.86     | 2000    |

> ✅ The model achieves **86% accuracy** in gender prediction, with balanced precision and recall across classes.

### **2. Age Prediction**

* **Mean Absolute Error (MAE):** 11.14 years
* On average, the model's predicted age is **off by ~11 years**.

> ⚡ Potential improvements: More data, advanced architectures (ResNet, EfficientNet), or age range grouping.

---

## 🌐 Deployment

This project is deployed on Hugging Face Spaces for **instant online usage**:

[ChronoVis – Hugging Face Space](https://huggingface.co/spaces/SabihaKhan/ChronoVis)

---

## 💡 Applications

* **Customer Analytics:** Estimate age and gender in retail or service environments.
* **Healthcare:** Assist in age-based diagnostics or patient management.
* **Interactive Interfaces:** Create personalized experiences in apps and kiosks.
* **Educational Use:** Demonstrates a full pipeline of deep learning from dataset to deployment.

---

## 🔥 Highlights

* Full end-to-end project from **data preprocessing → model training → deployment**.
* Efficient memory usage by limiting dataset to 10,000 images.
* Clean, modular, and scalable code ready for integration into larger ML pipelines.

---

## 👩‍💻 Author

**Sabiha Khan** – Aspiring Machine Learning Engineer | Data Enthusiast | UI/UX & Digital Marketing

* [LinkedIn](https://www.linkedin.com/in/sabiha-shehzadi)
* [GitHub](https://github.com/SabihaKhan)
* Email: [shehzadisabiha3425@gmail.com](mailto:shehzadisabiha3425@gmail.com)

---

## ⭐ Feedback

If you find this project useful or interesting, please ⭐ the repo and share your feedback! Contributions are welcome.


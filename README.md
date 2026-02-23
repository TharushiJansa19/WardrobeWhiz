![Wardrobe Whiz Banner](frontend/assets/poster.png)

# 👕 Wardrobe Whiz — AI Smart Wardrobe Management System

Wardrobe Whiz is an AI-powered wardrobe organization and outfit recommendation system that helps users digitally manage their clothing and discover matching outfits intelligently.

The system uses computer vision and machine learning to automatically detect clothing items, identify similarities, prevent duplicates, and recommend combinations based on context such as occasion and season.

---

## 🚀 Key Features

### 🧠 Intelligent Clothing Detection
- Uses **Transformer-based semantic segmentation (SegFormer)** to detect clothing regions from images
- Automatically separates clothing items from background
- Enables accurate classification and cataloging

### 🔍 Similarity & Duplicate Detection
- Generates feature embeddings using **Google Gemini models**
- Compares clothing items using vector similarity search
- Detects duplicate or visually similar items inside wardrobe

### 👗 Outfit Recommendation Engine
Hybrid recommendation system combining:

- Content-based filtering (color, type, style)
- Context-aware suggestions (occasion, season, usage)
- Visual similarity matching
- Personalized outfit suggestions

### 🔎 Smart Search
Users can search wardrobe using natural text queries:

> “black casual shirt for evening”  
> “formal outfit for interview”  
> “summer beach wear”

---

## 🏗 System Architecture

### Frontend
- React Native mobile application

### Backend
- FastAPI / Flask API services
- Image processing pipeline
- Recommendation engine

### AI / ML Stack
- PyTorch
- TensorFlow
- Transformers
- SegFormer model
- Gemini embeddings
- LlamaIndex retrieval

### Database
- MongoDB (user & item storage)
- Vector database (ChromaDB / Pinecone)

---

## 📊 Evaluation Metrics

The model performance was validated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC

The segmentation model achieved strong performance in clothing boundary detection and enabled reliable recommendation quality.

---

## 💡 Problem It Solves

Most people:

- Forget what clothes they own
- Buy duplicates
- Struggle to match outfits
- Waste time choosing what to wear

Wardrobe Whiz converts a physical wardrobe into an intelligent digital assistant that:

✔ Organizes clothes automatically  
✔ Suggests outfits instantly  
✔ Reduces duplicate purchases  
✔ Saves time daily  

---

## 🧪 Research Contributions

- Applied Transformer-based segmentation in a consumer wardrobe application
- Combined visual similarity search with recommendation systems
- Demonstrated practical AI usage in everyday lifestyle management
- Built an end-to-end AI product (not just a model)

---

## 🛠 Tech Stack

• Python • FastAPI • Flask • React Native
• PyTorch • TensorFlow • OpenCV • Transformers
• MongoDB • Pinecone • ChromaDB • LlamaIndex

---

## 👤 Author
**Tharushika Jansa**  
Frontend Developer | Software Engineer | AI Enthusiast

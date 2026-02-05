# 🌿 MangroveGPT: Real-Time Multi-Modal Multi-Agent Mangrove Monitoring with Agentic RAG

MangroveGPT is an AI-powered system that monitors mangrove health in real time by integrating multi-modal Earth observation data. It combines time series forecasting, satellite imagery analysis, and natural language understanding to support environmental research and decision-making.
---

## 🚀 Key Features

- **Real-Time Forecasting**: Predicts mangrove health using environmental station data and an XGBoost time series model trained on NDVI (Normalized Difference Vegetation Index).
- **Multi-Modal Query Routing**: Automatically classifies and routes textual and image-based user queries.
- **Satellite Image Understanding**: Uses a fine-tuned CLIP vision-language model and captioning pipeline to describe satellite images and route them appropriately.
- **RAG Pipeline**: Supports retrieval-augmented generation (RAG) using OpenAI + Pinecone + LangChain to answer research-based questions.
- **Streamlit App**: Deployed with an easy-to-use web interface for users from any background—no coding required.

---

## 🧠 How It Works

1. **User Query** → Interpreted using a structured LLM.
2. **Query Type Classification**:
   - Text queries → Forecast or Research pipeline
   - Image queries → CLIP-based captioning → Text pipeline
3. **Forecast Pipeline**:
   - Fetches real-time data (wind, water level, NDVI)
   - Builds a 7-week lagged feature vector
   - Predicts NDVI using XGBoost
   - Summarizes with LLM
4. **Research Pipeline**:
   - Uses LangChain and Pinecone to answer general mangrove questions
5. **Interface**:
   - Accessible through Streamlit for real-time usage.

---

## 🛠 Tech Stack

- **LLMs**: OpenAI (ChatGPT / Gemini), LangChain
- **Data APIs**: NOAA, Google Earth Engine, Google Maps
- **ML**: XGBoost (time series forecasting)
- **Vision**: CLIP (fine-tuned on mangrove imagery)
- **Search**: Pinecone
- **Frontend**: Streamlit
- **Framework**: LangGraph (multi-agent pipeline)

---

## 📊 Model Performance

- **MAPE**: 2% (Mean Absolute Percentage Error)
- Roughly equivalent to **98% accuracy** for NDVI forecasting — extremely reliable for environmental insights.

---

## 💻 Getting Started

```bash
# Clone the repo
git clone https://github.com/GilbertHarijanto/MangroveAgent.git
cd mangrovegpt

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI backend
uvicorn backend.main:app --reload

# In a separate terminal, launch React frontend
cd frontend
npm install
npm run dev

# Optional: legacy Streamlit app
streamlit run app.py

"""
Natural Language Inference - Streamlit App
File: app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NLI Model - Curriculum Meta-Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    .example-box {
        padding: 1rem;
        border-radius: 8px;
        background: #e8f4f8;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-box:hover {
        background: #d1e7f0;
        transform: translateX(5px);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class Config:
    model_name = 'microsoft/mdeberta-v3-base'
    num_labels = 3

class MetaNLIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.model_name,
            num_labels=Config.num_labels
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# ============================================================================
# LOAD MODEL (Cached)
# ============================================================================

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    
    with st.spinner("🔄 Downloading model from Hugging Face..."):
        checkpoint_path = hf_hub_download(
            repo_id="hmzi67/mdeberta-xnli",
            filename="meta_episode_400.pt",
            cache_dir="./model_cache"
        )
    
    with st.spinner("🔄 Loading model into memory..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        model = MetaNLIModel().to(device)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
    
    return model, tokenizer, device

# Load model
model, tokenizer, device = load_model()
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_nli(premise, hypothesis):
    """Make prediction"""
    
    encoded = tokenizer(
        premise,
        hypothesis,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_label = probs.argmax()
    
    return label_map[pred_label], probs

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_probability_chart(probs):
    """Create interactive bar chart"""
    
    labels = ['Entailment', 'Neutral', 'Contradiction']
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probs * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Relationship",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 📚 About")
    st.markdown("""
    This model uses **Curriculum Meta-Learning** (MAML) for Natural Language Inference.
    
    ### 🎯 What it does:
    Predicts the logical relationship between two sentences.
    
    ### 🏆 Model Details:
    - **Base**: mDeBERTa-v3-base
    - **Method**: Curriculum Meta-Learning
    - **Training**: English + Meta-learning
    - **Languages**: 15 XNLI languages
    """)
    
    st.markdown("---")
    
    st.markdown("## 🔗 Links")
    st.markdown("""
    - [Model Repository](https://huggingface.co/hmzi67/mdeberta-xnli)
    - [GitHub](https://github.com/hmzi67)
    - [Paper](https://arxiv.org/abs/your-paper)
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "85.4%")
    with col2:
        st.metric("Languages", "15")
    
    st.markdown("---")
    st.markdown("### ⚡ Device")
    st.info(f"Running on: **{device}**")

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🤖 Natural Language Inference</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Curriculum Meta-Learning for Zero-Shot Cross-Lingual Transfer</p>', unsafe_allow_html=True)

# Info section
with st.expander("📖 What is Natural Language Inference (NLI)?", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### ✅ Entailment
        The hypothesis **logically follows** from the premise.
        
        **Example:**
        - Premise: "A man is playing guitar"
        - Hypothesis: "A musician is performing"
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ Neutral
        The hypothesis **might be true** given the premise.
        
        **Example:**
        - Premise: "A person is riding a horse"
        - Hypothesis: "The person is training"
        """)
    
    with col3:
        st.markdown("""
        ### ❌ Contradiction
        The hypothesis **contradicts** the premise.
        
        **Example:**
        - Premise: "It's raining heavily"
        - Hypothesis: "The weather is sunny"
        """)

st.markdown("---")

# Main input section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Input")
    
    premise = st.text_area(
        "**Premise** (Factual statement)",
        placeholder="Enter the premise sentence here...",
        height=120,
        key="premise"
    )
    
    hypothesis = st.text_area(
        "**Hypothesis** (Statement to test)",
        placeholder="Enter the hypothesis sentence here...",
        height=120,
        key="hypothesis"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        predict_button = st.button("🔮 Predict Relationship", type="primary")
    with col_btn2:
        clear_button = st.button("🗑️ Clear")

with col2:
    st.markdown("### 💡 Try Examples")
    
    examples = [
        {
            "premise": "A person on a horse jumps over a broken down airplane.",
            "hypothesis": "A person is training his horse for a competition.",
            "expected": "Neutral"
        },
        {
            "premise": "A man is playing guitar on stage.",
            "hypothesis": "A musician is performing.",
            "expected": "Entailment"
        },
        {
            "premise": "The cat is sleeping on the couch.",
            "hypothesis": "The dog is sleeping.",
            "expected": "Contradiction"
        },
        {
            "premise": "It's raining heavily outside.",
            "hypothesis": "The weather is sunny.",
            "expected": "Contradiction"
        },
        {
            "premise": "Scientists discovered a new planet.",
            "hypothesis": "Astronomers found a new celestial body.",
            "expected": "Entailment"
        }
    ]
    
    for i, ex in enumerate(examples):
        with st.container():
            if st.button(f"📌 Example {i+1}: {ex['expected']}", key=f"ex_{i}"):
                st.session_state.premise = ex['premise']
                st.session_state.hypothesis = ex['hypothesis']
                st.rerun()
            
            with st.expander(f"View Example {i+1}"):
                st.markdown(f"**Premise:** {ex['premise']}")
                st.markdown(f"**Hypothesis:** {ex['hypothesis']}")
                st.markdown(f"**Expected:** {ex['expected']}")

# Clear button logic
if clear_button:
    st.session_state.premise = ""
    st.session_state.hypothesis = ""
    st.rerun()

# Prediction logic
if predict_button:
    if not premise.strip() or not hypothesis.strip():
        st.error("⚠️ Please enter both premise and hypothesis!")
    else:
        # Progress bar for dramatic effect
        with st.spinner("🤔 Analyzing relationship..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Make prediction
            prediction, probs = predict_nli(premise, hypothesis)
            confidence = probs.max() * 100
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Results")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction</h3>
                <h2>{prediction.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <h2>{confidence:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            emoji_map = {
                'entailment': '✅',
                'neutral': '⚖️',
                'contradiction': '❌'
            }
            st.markdown(f"""
            <div class="metric-card">
                <h3>Status</h3>
                <h2>{emoji_map.get(prediction, '🤔')}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Probability Distribution")
            fig = create_probability_chart(probs)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Detailed Breakdown")
            st.markdown(f"""
            <div class="prediction-box">
                <h4>Input Texts:</h4>
                <p><strong>Premise:</strong> {premise}</p>
                <p><strong>Hypothesis:</strong> {hypothesis}</p>
                <hr>
                <h4>Probabilities:</h4>
                <ul>
                    <li><strong>Entailment:</strong> {probs[0]*100:.2f}%</li>
                    <li><strong>Neutral:</strong> {probs[1]*100:.2f}%</li>
                    <li><strong>Contradiction:</strong> {probs[2]*100:.2f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ using Streamlit & Hugging Face</p>
    <p>Model: <a href='https://huggingface.co/hmzi67/mdeberta-xnli'>hmzi67/mdeberta-xnli</a></p>
</div>
""", unsafe_allow_html=True)
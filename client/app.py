"""
Natural Language Inference - Streamlit App (Professional Design)
File: app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NLI Model - Curriculum Meta-Learning",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'premise' not in st.session_state:
    st.session_state.premise = ""
if 'hypothesis' not in st.session_state:
    st.session_state.hypothesis = ""
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# ============================================================================
# PROFESSIONAL CSS - BLACK & ORANGE THEME
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark theme with proper contrast */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
        background-color: #0a0a0a;
    }
    
    /* Professional header */
    .hero-section {
        text-align: center;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 2px solid #ff6b35;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff6b35 0%, #ff8c42 100%);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Professional card styling */
    .card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #2d2d2d;
        height: 100%;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ff6b35;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Text areas with proper visibility */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        border: 2px solid #3d3d3d !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        min-height: 100px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.15) !important;
        background-color: #1a1a1a !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #666666 !important;
    }
    
    /* Professional buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
        color: #ffffff;
        border: none;
        padding: 0.9rem 2rem;
        font-size: 1rem;
        border-radius: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
        background: linear-gradient(135deg, #ff8c42 0%, #ff6b35 100%);
    }
    
    /* Result cards */
    .result-metric {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        border: 2px solid #ff6b35;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .result-metric:hover {
        transform: translateY(-5px);
        border-color: #ff8c42;
    }
    
    .result-metric-label {
        font-size: 0.85rem;
        color: #b0b0b0;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .result-metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ff6b35;
        margin: 0;
    }
    
    /* Chart container */
    .chart-container {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2d2d2d;
    }
    
    /* Breakdown box */
    .breakdown-box {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #ff6b35;
        border: 1px solid #2d2d2d;
    }
    
    .breakdown-box h4 {
        color: #ff6b35;
        margin-top: 0;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 1rem;
    }
    
    .breakdown-item {
        display: flex;
        justify-content: space-between;
        padding: 0.8rem 0;
        border-bottom: 1px solid #2d2d2d;
        color: #ffffff;
    }
    
    .breakdown-item:last-child {
        border-bottom: none;
    }
    
    .breakdown-label {
        color: #b0b0b0;
        font-weight: 500;
    }
    
    .breakdown-value {
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Example cards */
    .example-card {
        background: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #ffffff;
    }
    
    .example-card:hover {
        border-color: #ff6b35;
        transform: translateX(5px);
        background: #1a1a1a;
    }
    
    .example-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-entailment { 
        background: #28a745; 
        color: #ffffff; 
    }
    .badge-neutral { 
        background: #ffc107; 
        color: #1a1a1a; 
    }
    .badge-contradiction { 
        background: #dc3545; 
        color: #ffffff; 
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: #1a1a1a;
        border-radius: 12px;
        border: 2px dashed #3d3d3d;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        color: #ff6b35;
    }
    
    .empty-state-text {
        font-size: 1.2rem;
        color: #b0b0b0;
        font-weight: 600;
    }
    
    .empty-state-subtext {
        font-size: 0.9rem;
        color: #666666;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0f0f0f;
        border-right: 2px solid #ff6b35;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ff6b35 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #ff8c42 !important;
    }
    
    /* Info box */
    .info-box {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #2d2d2d;
    }
    
    .info-box strong {
        color: #ff6b35 !important;
    }
    
    /* Metrics in sidebar */
    .sidebar-metric {
        text-align: center;
        padding: 1rem;
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #2d2d2d;
    }
    
    .sidebar-metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ff6b35;
    }
    
    .sidebar-metric-label {
        font-size: 0.85rem;
        color: #b0b0b0;
        margin-top: 0.3rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666666;
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 1px solid #2d2d2d;
    }
    
    .footer a {
        color: #ff6b35;
        text-decoration: none;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Fix metric display */
    [data-testid="stMetricValue"] {
        color: #ff6b35 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
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
    with st.spinner("Loading AI model..."):
        checkpoint_path = hf_hub_download(
            repo_id="hmzi67/mdeberta-xnli",
            filename="meta_episode_400.pt",
            cache_dir="./model_cache"
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        model = MetaNLIModel().to(device)
        
        # Load checkpoint with pickle to handle custom classes
        try:
            # First try with weights_only=False to allow custom objects
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
            # Fallback: try loading with pickle
            import pickle
            import sys
            
            # Add current module to sys.modules to help pickle find classes
            sys.modules['__main__'].Config = Config
            sys.modules['__main__'].MetaNLIModel = MetaNLIModel
            
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False, pickle_module=pickle)
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict into model
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            st.warning(f"Loading with strict=False due to: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
    
    return model, tokenizer, device

model, tokenizer, device = load_model()
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_nli(premise, hypothesis):
    """Make prediction"""
    encoded = tokenizer(
        premise, hypothesis,
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

def create_professional_chart(probs):
    """Create professional probability chart"""
    labels = ['Entailment', 'Neutral', 'Contradiction']
    colors = ['#28a745', '#ffc107', '#dc3545']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=probs * 100,
            marker=dict(
                color=colors,
                line=dict(color='#1a1a1a', width=2)
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='outside',
            textfont=dict(size=14, color='#ffffff', family='Inter'),
        )
    ])
    
    fig.update_layout(
        title=dict(text="Confidence Distribution", font=dict(size=16, color='#ff6b35', family='Inter')),
        xaxis=dict(
            title="", 
            tickfont=dict(size=12, color='#ffffff'),
            gridcolor='#2d2d2d'
        ),
        yaxis=dict(
            title="Confidence (%)", 
            range=[0, max(probs)*110], 
            tickfont=dict(size=12, color='#ffffff'),
            gridcolor='#2d2d2d'
        ),
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
    )
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### MODEL INFORMATION")
    
    st.markdown("""
    <div class="info-box">
    <strong>Architecture:</strong> mDeBERTa-v3-base<br>
    <strong>Method:</strong> Curriculum Meta-Learning<br>
    <strong>Training:</strong> MAML on XNLI<br>
    <strong>Languages:</strong> 15 Cross-lingual
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### PERFORMANCE METRICS")
    
    # Row 1 → Accuracy
    with st.container():
        st.markdown("""
        <div class='sidebar-metric'>
            <div class='sidebar-metric-value'>80.41%</div>
            <div class='sidebar-metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2 → Languages
    with st.container():
        st.markdown("""
        <div class='sidebar-metric'>
            <div class='sidebar-metric-value'>15</div>
            <div class='sidebar-metric-label'>Languages</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    
    st.markdown("### WHAT IS NLI?")
    st.markdown("""
    <div class="info-box">
    Natural Language Inference determines the logical relationship between two sentences:
    <br><br>
    <strong>Entailment:</strong> Hypothesis follows from premise<br>
    <strong>Neutral:</strong> Hypothesis might be true<br>
    <strong>Contradiction:</strong> Hypothesis conflicts with premise
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### RESOURCES")
    st.markdown("""
    - [Model Repository](https://huggingface.co/hmzi67/mdeberta-xnli)
    - [GitHub Profile](https://github.com/hmzi67)
    - [Documentation](https://huggingface.co/spaces/hmzi67/entailment-xnli-nlp)
    """)
    
    st.markdown(f"""
    <div class="info-box" style="margin-top: 1rem;">
    <strong>Device:</strong> {device}<br>
    <strong>Status:</strong> Ready
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Natural Language Inference</div>
    <div class="hero-subtitle">Powered by Curriculum Meta-Learning • Zero-Shot Cross-Lingual Transfer</div>
</div>
""", unsafe_allow_html=True)

# Main Layout - Two Columns
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Input Sentences</div>', unsafe_allow_html=True)
    
    premise = st.text_area(
        "Premise (Factual Statement)",
        value=st.session_state.premise,
        placeholder="Enter your premise sentence here...",
        help="The premise is a factual statement that serves as the basis for inference",
        key="premise_input"
    )
    
    hypothesis = st.text_area(
        "Hypothesis (Statement to Test)",
        value=st.session_state.hypothesis,
        placeholder="Enter your hypothesis sentence here...",
        help="The hypothesis is the statement whose relationship to the premise will be determined",
        key="hypothesis_input"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        predict_button = st.button("Analyze Relationship", type="primary")
    with col_btn2:
        clear_button = st.button("Clear")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Examples Section
    st.markdown('<div class="card animate-in" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Example Inputs</div>', unsafe_allow_html=True)
    
    examples = [
        ("A person on a horse jumps over a broken airplane.", "A person is training his horse for competition.", "neutral"),
        ("A man is playing guitar on stage.", "A musician is performing.", "entailment"),
        ("The cat is sleeping on the couch.", "The dog is sleeping.", "contradiction"),
        ("It's raining heavily outside.", "The weather is sunny and clear.", "contradiction"),
        ("Scientists discovered a new planet.", "Astronomers found a celestial body.", "entailment"),
    ]
    
    for i, (p, h, expected) in enumerate(examples):
        badge_class = f"badge-{expected}"
        if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state.premise = p
            st.session_state.hypothesis = h
            st.session_state.show_results = False
            st.rerun()
        
        with st.expander(f"Preview: {expected.title()}", expanded=False):
            st.markdown(f"""
            <div style="padding: 0.5rem;">
                <span class="example-badge {badge_class}">{expected.upper()}</span>
                <p style="color: #ffffff;"><strong style="color: #ff6b35;">Premise:</strong> {p}</p>
                <p style="color: #ffffff;"><strong style="color: #ff6b35;">Hypothesis:</strong> {h}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_results:
    if clear_button:
        st.session_state.premise = ""
        st.session_state.hypothesis = ""
        st.session_state.show_results = False
        st.rerun()
    
    if predict_button:
        if not premise.strip() or not hypothesis.strip():
            st.error("Please enter both premise and hypothesis!")
        else:
            with st.spinner("Analyzing relationship..."):
                prediction, probs = predict_nli(premise, hypothesis)
                confidence = probs.max() * 100
                st.session_state.show_results = True
                st.session_state.prediction = prediction
                st.session_state.probs = probs
                st.session_state.confidence = confidence
    
    # Results Display
    if st.session_state.show_results:
        st.markdown('<div class="animate-in">', unsafe_allow_html=True)
        
        # Metrics Row
        st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        # Prediction Row (single column)
        with st.container():
            pred_col = st.columns([1])[0]
            pred_col.markdown(f"""
            <div class="result-metric">
                <div class="result-metric-label">Prediction</div>
                <div class="result-metric-value">{st.session_state.prediction.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Next Row: Confidence + Status
        with st.container():
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown(f"""
                <div class="result-metric">
                    <div class="result-metric-label">Confidence</div>
                    <div class="result-metric-value">{st.session_state.confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
            with col2:
                status_text = {
                    'entailment': 'VALID',
                    'neutral': 'UNCERTAIN',
                    'contradiction': 'INVALID'
                }
                st.markdown(f"""
                <div class="result-metric">
                    <div class="result-metric-label">Status</div>
                    <div class="result-metric-value">{status_text.get(st.session_state.prediction, 'UNKNOWN')}</div>
                </div>
                """, unsafe_allow_html=True)

        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_professional_chart(st.session_state.probs)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Breakdown
        st.markdown('<div class="breakdown-box" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<h4>Detailed Analysis</h4>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="breakdown-item">
            <span class="breakdown-label">Entailment</span>
            <span class="breakdown-value" style="color: #28a745;">{st.session_state.probs[0]*100:.2f}%</span>
        </div>
        <div class="breakdown-item">
            <span class="breakdown-label">Neutral</span>
            <span class="breakdown-value" style="color: #ffc107;">{st.session_state.probs[1]*100:.2f}%</span>
        </div>
        <div class="breakdown-item">
            <span class="breakdown-label">Contradiction</span>
            <span class="breakdown-value" style="color: #dc3545;">{st.session_state.probs[2]*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Empty State
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">▶</div>
            <div class="empty-state-text">Ready to Analyze</div>
            <p class="empty-state-subtext">Enter sentences and click "Analyze Relationship"</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Code with ❤️ by HMZI67
    <a href="https://huggingface.co/hmzi67/mdeberta-xnli">Model: hmzi67/mdeberta-xnli</a>
</div>
""", unsafe_allow_html=True)
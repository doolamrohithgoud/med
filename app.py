"""
app.py — Health Information RAG Assistant (Streamlit UI)
Main application entry point. Implements a clean, professional health assistant UI.
"""

import streamlit as st
from PIL import Image
import io
import os
from dotenv import load_dotenv

from rag_engine import (
    build_vector_store,
    retrieve_context,
    is_health_query,
    is_critical_condition,
    generate_response,
)
from image_analyzer import analyze_medicine_image

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthRAG — AI Health Information Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 50%, #0f1729 100%);
    min-height: 100vh;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60d0ff, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.main-header p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
}

/* Disclaimer banner */
.disclaimer-banner {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid rgba(234, 179, 8, 0.3);
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 1.5rem;
    color: #fbbf24;
    font-size: 0.82rem;
    text-align: center;
}

/* Critical emergency banner */
.critical-banner {
    background: rgba(239, 68, 68, 0.15);
    border: 2px solid rgba(239, 68, 68, 0.5);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 1rem 0;
    color: #f87171;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { border-color: rgba(239, 68, 68, 0.5); }
    50% { border-color: rgba(239, 68, 68, 0.9); }
}

/* Domain rejection banner */
.rejection-banner {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 1rem 0;
    color: #a5b4fc;
    font-size: 0.95rem;
    text-align: center;
}

/* Response cards */
.response-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.response-card:hover {
    border-color: rgba(96, 208, 255, 0.3);
}
.response-card h4 {
    color: #60d0ff;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.response-card p {
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.65;
    margin: 0;
    white-space: pre-wrap;
}

/* Confidence badge */
.confidence-high { color: #34d399; border-color: rgba(52, 211, 153, 0.4); background: rgba(52, 211, 153, 0.08); }
.confidence-medium { color: #fbbf24; border-color: rgba(251, 191, 36, 0.4); background: rgba(251, 191, 36, 0.08); }
.confidence-low { color: #f87171; border-color: rgba(248, 113, 113, 0.4); background: rgba(248, 113, 113, 0.08); }
.confidence-badge {
    display: inline-block;
    border-radius: 20px;
    border: 1px solid;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 4px;
}

/* Medicine image result card */
.med-card {
    background: rgba(167, 139, 250, 0.06);
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 1rem;
}
.med-card h4 {
    color: #a78bfa;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.med-card p {
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.65;
    margin: 0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}

/* Input area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: rgba(96, 208, 255, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(96, 208, 255, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

/* File uploader */
.stFileUploader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(167,139,250,0.3) !important;
    border-radius: 12px !important;
}

/* History item */
.history-item {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 3px solid rgba(96, 208, 255, 0.4);
    color: #94a3b8;
    font-size: 0.82rem;
    cursor: pointer;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.2rem 0;
}

/* Spinner override */
.stSpinner > div {
    border-top-color: #60d0ff !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
def init_session():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "vs_ready" not in st.session_state:
        st.session_state.vs_ready = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "last_image_result" not in st.session_state:
        st.session_state.last_image_result = None

init_session()


# ─── Vector Store Initialization ─────────────────────────────────────────────
def get_vector_store():
    if not st.session_state.vs_ready:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your_google_api_key_here":
            st.error("⚠️ Please add your GOOGLE_API_KEY to the `.env` file and restart the app.")
            st.stop()
        with st.spinner("🔬 Initializing health knowledge base (first run may take ~30s)..."):
            try:
                st.session_state.vector_store = build_vector_store()
                st.session_state.vs_ready = True
            except Exception as e:
                st.error(f"Failed to initialize knowledge base: {e}")
                st.stop()
    return st.session_state.vector_store


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:2.5rem; margin-bottom:0.4rem;">🏥</div>
        <div style="color:#60d0ff; font-weight:700; font-size:1.15rem;">HealthRAG</div>
        <div style="color:#64748b; font-size:0.78rem;">AI Health Information Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.selectbox(
        "🎯 Analysis Mode",
        ["Symptom Analysis", "Medicine Image Analysis"],
        help="Choose what you want to analyze",
    )

    st.markdown("---")

    if mode == "Symptom Analysis":
        st.markdown("**Patient Context**")
        age_group = st.selectbox(
            "Age Group",
            ["Child (under 12)", "Teenager (12–17)", "Adult (18–64)", "Elderly (65+)"],
            index=2,
        )
        severity = st.selectbox(
            "Symptom Severity",
            ["Mild", "Moderate", "Severe"],
        )
        duration = st.text_input(
            "Duration of Symptoms",
            placeholder="e.g., 2 days, 1 week",
        )

    st.markdown("---")

    # History panel
    st.markdown("**📋 Query History**")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-6:])):
            q_preview = item["query"][:50] + ("..." if len(item["query"]) > 50 else "")
            st.markdown(f"""
            <div class="history-item">
                <strong style="color:#cbd5e1;">#{len(st.session_state.history)-i}</strong> {q_preview}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#475569; font-size:0.8rem;">No queries yet.</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.session_state.last_response = None
        st.session_state.last_image_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#475569; font-size:0.72rem; text-align:center; line-height:1.6;">
        <strong style="color:#64748b;">⚠️ Disclaimer</strong><br>
        For informational purposes only.<br>
        Not a substitute for professional<br>
        medical advice or diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 HealthRAG Assistant</h1>
    <p>AI-powered health information retrieval — grounded in verified medical references</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>Medical Disclaimer:</strong> This assistant provides general health information only.
    It does NOT diagnose diseases, prescribe medicines, or replace a licensed medical professional.
    For any medical concern, always consult a qualified doctor or healthcare provider.
</div>
""", unsafe_allow_html=True)


# ─── SYMPTOM ANALYSIS MODE ────────────────────────────────────────────────────
if mode == "Symptom Analysis":
    st.markdown("### 💬 Describe Your Symptoms")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_area(
            "Symptoms and Health Question",
            placeholder=(
                "Example: I have had a persistent headache and mild fever for 2 days. "
                "I also have a runny nose and feel tired. What could this indicate?"
            ),
            height=130,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", key="analyze_symptoms", use_container_width=True)
        st.info(
            f"**Mode:** Symptom Analysis\n\n"
            f"**Age:** {age_group}\n\n"
            f"**Severity:** {severity}\n\n"
            f"**Duration:** {duration or 'Not specified'}"
        )

    if analyze_btn and user_query.strip():
        vs = get_vector_store()

        # ── Critical condition check ──
        if is_critical_condition(user_query):
            st.markdown("""
            <div class="critical-banner">
                🚨 <strong>CRITICAL SITUATION DETECTED</strong><br><br>
                This situation may be serious. Please consult a licensed medical doctor
                or visit a healthcare facility <strong>immediately</strong>.<br><br>
                If this is an emergency — <strong>call emergency services now.</strong>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append({"query": user_query, "type": "critical"})

        # ── Severity override check ──
        elif severity == "Severe":
            st.markdown("""
            <div class="critical-banner">
                🚨 <strong>SEVERE SYMPTOMS REPORTED</strong><br><br>
                You have indicated that your symptoms are severe.
                Please consult a licensed medical doctor or visit a healthcare facility promptly.
                Do not delay seeking professional care.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append({"query": user_query, "type": "severe"})

        # ── Age group check (child/elderly) ──
        elif age_group in ["Child (under 12)"]:
            st.markdown("""
            <div class="critical-banner">
                🚨 <strong>PEDIATRIC CONCERN</strong><br><br>
                For children under 12 with any significant symptoms, it is strongly
                recommended to consult a pediatrician or licensed medical doctor directly.
                Do not rely solely on general health information tools.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append({"query": user_query, "type": "pediatric"})

        # ── Domain gate ──
        elif not is_health_query(user_query):
            st.markdown("""
            <div class="rejection-banner">
                🚫 <strong>Out-of-Scope Query</strong><br>
                I am designed only for health-related assistance. I cannot help with this request.
            </div>
            """, unsafe_allow_html=True)

        else:
            with st.spinner("🔬 Retrieving medical references and generating response..."):
                context = retrieve_context(user_query, vs)
                response = generate_response(
                    query=user_query,
                    context=context,
                    age_group=age_group,
                    severity=severity,
                    duration=duration or "Not specified",
                )
                st.session_state.last_response = response
                st.session_state.history.append({"query": user_query, "type": "symptom"})

    elif analyze_btn and not user_query.strip():
        st.warning("Please describe your symptoms before clicking Analyze.")

    # ── Display last response ──
    if st.session_state.last_response:
        resp = st.session_state.last_response
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📋 Health Information Response")

        if resp.get("success") and resp.get("sections"):
            secs = resp["sections"]

            # Section map
            section_map = [
                ("query_understanding", "🎯 Query Understanding"),
                ("evidence_summary", "📚 Retrieved Medical Evidence"),
                ("observation", "🔍 Symptom Observation"),
                ("possible_associations", "🔗 Possible Associations (Non-Diagnostic)"),
                ("safe_guidance", "✅ Safe Guidance"),
            ]

            for key, title in section_map:
                content = secs.get(key, "").strip()
                if content:
                    st.markdown(f"""
                    <div class="response-card">
                        <h4>{title}</h4>
                        <p>{content}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Confidence
            conf_text = secs.get("confidence", "Medium").strip()
            conf_lower = conf_text.lower()
            if "high" in conf_lower:
                badge_class = "confidence-high"
                badge_label = "High Confidence"
            elif "medium" in conf_lower or "moderate" in conf_lower:
                badge_class = "confidence-medium"
                badge_label = "Medium Confidence"
            else:
                badge_class = "confidence-low"
                badge_label = "Low Confidence"

            st.markdown(f"""
            <div class="response-card">
                <h4>📊 Confidence Level</h4>
                <p>{conf_text}</p>
                <span class="confidence-badge {badge_class}">{badge_label}</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="response-card">
                <h4>⚠️ Response</h4>
                <p>{resp.get('raw', 'No response generated.')}</p>
            </div>
            """, unsafe_allow_html=True)

        # Doctor referral footer
        st.markdown("""
        <div style="background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.2);
                    border-radius:12px; padding:14px 18px; margin-top:1rem; text-align:center;
                    color:#6ee7b7; font-size:0.85rem;">
            💡 <strong>Remember:</strong> This information is for educational purposes only.
            When in doubt, always consult a licensed medical professional.
        </div>
        """, unsafe_allow_html=True)


# ─── MEDICINE IMAGE ANALYSIS MODE ────────────────────────────────────────────
else:
    st.markdown("### 💊 Medicine / Tablet Image Analysis")

    st.markdown("""
    <div style="background:rgba(167,139,250,0.07); border:1px solid rgba(167,139,250,0.2);
                border-radius:12px; padding:14px 18px; margin-bottom:1.2rem;
                color:#c4b5fd; font-size:0.85rem; line-height:1.6;">
        📷 <strong>Upload a clear image of a medicine tablet, capsule, or packaging.</strong><br>
        The system will attempt to identify the medicine from visible text, imprints, or packaging only.
        Identification is only confirmed when text is clearly readable.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Medicine / Tablet Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            try:
                pil_img = Image.open(uploaded_file)
                st.image(pil_img, caption="Uploaded Medicine Image", use_container_width=True)
            except Exception:
                st.error("Could not display the image.")

        with col2:
            analyze_med_btn = st.button("🔬 Analyze Medicine", key="analyze_medicine", use_container_width=True)

            st.markdown("""
            <div style="color:#94a3b8; font-size:0.82rem; line-height:1.7; margin-top:1rem;">
                <strong style="color:#a78bfa;">Analysis covers:</strong><br>
                ✓ Visible imprint/label text<br>
                ✓ Medicine identification (if readable)<br>
                ✓ Common use information<br>
                ✓ Safety notes<br>
                ✓ Expiry date reminder
            </div>
            """, unsafe_allow_html=True)

        if analyze_med_btn:
            # Re-read file bytes
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()

            with st.spinner("🔬 Analyzing medicine image..."):
                result = analyze_medicine_image(image_bytes)
                st.session_state.last_image_result = result
                st.session_state.history.append({
                    "query": f"[Image] {uploaded_file.name}",
                    "type": "medicine_image",
                })

    # ── Display image analysis result ──
    if st.session_state.last_image_result:
        result = st.session_state.last_image_result
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 💊 Medicine Analysis Result")

        if not result.get("success"):
            st.markdown(f"""
            <div class="med-card">
                <h4>⚠️ Analysis Error</h4>
                <p>{result.get('message', 'Analysis failed.')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Medicine Identification
            identified = result.get("identified_medicine")
            if identified and "cannot be reliably identified" not in identified.lower():
                st.markdown(f"""
                <div class="med-card">
                    <h4>💊 Medicine Identification</h4>
                    <p>{identified}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="med-card">
                    <h4>💊 Medicine Identification</h4>
                    <p>⚠️ <strong>The medicine cannot be reliably identified from this image.</strong><br>
                    Please check the packaging label or consult a licensed pharmacist for proper identification.</p>
                </div>
                """, unsafe_allow_html=True)

            # Common Use
            common_use = result.get("common_use", "").strip()
            if common_use:
                st.markdown(f"""
                <div class="med-card">
                    <h4>📋 Common Use</h4>
                    <p>{common_use}</p>
                </div>
                """, unsafe_allow_html=True)

            # Safety Notes
            safety = result.get("safety_notes", "").strip()
            if safety:
                st.markdown(f"""
                <div class="med-card">
                    <h4>⚠️ Safety Notes</h4>
                    <p>{safety}</p>
                </div>
                """, unsafe_allow_html=True)

            # Expiry Reminder
            expiry = result.get("expiry_reminder", "").strip()
            st.markdown(f"""
            <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25);
                        border-radius:12px; padding:14px 18px; margin-bottom:1rem;">
                <h4 style="color:#fbbf24; font-size:0.82rem; font-weight:600;
                           text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
                    📅 Expiry & Safety Check
                </h4>
                <p style="color:#cbd5e1; font-size:0.92rem; line-height:1.65; margin:0;">
                    {expiry or "Always check the expiry date printed on medicine packaging before use. Do not use expired medicines. Consult a pharmacist if unsure."}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence
            conf = result.get("confidence", "Low").strip()
            conf_lower = conf.lower()
            if "high" in conf_lower:
                badge_class = "confidence-high"
            elif "medium" in conf_lower or "moderate" in conf_lower:
                badge_class = "confidence-medium"
            else:
                badge_class = "confidence-low"

            st.markdown(f"""
            <div class="med-card">
                <h4>📊 Identification Confidence</h4>
                <p>{conf}</p>
                <span class="confidence-badge {badge_class}">
                    {"High" if "high" in conf_lower else "Medium" if "medium" in conf_lower else "Low"} Confidence
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Always show pharmacist reminder
        st.markdown("""
        <div style="background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.2);
                    border-radius:12px; padding:14px 18px; margin-top:0.5rem; text-align:center;
                    color:#6ee7b7; font-size:0.85rem;">
            💡 <strong>Always verify with a licensed pharmacist or doctor before taking any medicine.</strong>
            Never take an unidentified medicine. Check expiry date and follow packaging instructions.
        </div>
        """, unsafe_allow_html=True)

    elif mode == "Medicine Image Analysis" and not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#475569;">
            <div style="font-size:3.5rem; margin-bottom:1rem;">💊</div>
            <div style="font-size:1rem; color:#64748b;">
                Upload a medicine image using the file uploader above
            </div>
        </div>
        """, unsafe_allow_html=True)

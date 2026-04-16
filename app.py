# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from typing import Optional
from ripeness_database import FRESHNESS_DATA, FOOD_WASTE_FACTS
from datetime import datetime, timedelta


STAGE_LABEL_ALIASES = {
    "strawberry": {
        "fresh": "fresh",
        "spoiled": "spoiled",
        "unripe": "unripe",
        "with bruises fruit": "bruised_overripe",
    },
    "spinach": {
        "fresh": "fresh",
        "spoiled": "expired",
        "ageing": "aging",
    },
    "tomato": {
        "fresh": "fresh",
        "ripe": "fresh",
        "turning": "overripe_eatable",
        "spoiled": "spoiled",
        "overripe": "overripe_eatable",
        "immature": "unripe",
        "unripe": "unripe",
    },
    "banana": {
        "fresh": "fresh",
        "ripe": "medium",
        "overripe": "overripe_bread",
        "spoiled": "spoiled",
        "underripe": "underripe",
    },
    "avocado": {
        "fresh": "fresh",
        "ripe": "fresh",
        "mushy": "overripe_eatable",
        "spoiled": "spoiled",
        "unripe": "unripe",
        "almost ripe": "fresh",
    },
}


def normalize_freshness_stage(produce_name: str, raw_stage: str) -> Optional[str]:
    cleaned = raw_stage.strip().lower().replace("_", " ")
    cleaned = cleaned.replace("spoinach", "spinach").replace("unrripe", "unripe")
    prefix = f"{produce_name.lower()} "
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):]

    canonical = STAGE_LABEL_ALIASES.get(produce_name.lower(), {}).get(cleaned)
    if canonical:
        return canonical

    fallback = cleaned.replace(" ", "_")
    if fallback in FRESHNESS_DATA.get(produce_name, {}):
        return fallback
    return None


# Approximate impact factors for educational estimates (not exact per farm/source).
IMPACT_FACTORS = {
    "banana": {"kg_per_unit": 0.12, "co2e_kg_per_kg": 0.9, "water_l_per_kg": 790},
    "avocado": {"kg_per_unit": 0.17, "co2e_kg_per_kg": 2.5, "water_l_per_kg": 1980},
    "tomato": {"kg_per_unit": 0.12, "co2e_kg_per_kg": 1.1, "water_l_per_kg": 214},
    "strawberry": {"kg_per_unit": 0.018, "co2e_kg_per_kg": 1.4, "water_l_per_kg": 276},
    "spinach": {"kg_per_unit": 0.03, "co2e_kg_per_kg": 2.0, "water_l_per_kg": 292},
}

CAR_KGCO2_PER_KM = 0.192
PHONE_CHARGE_KGCO2 = 0.00822
SHOWER_LITERS = 65

# Page config
st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🥗",
    layout="wide"
)


# Load models
@st.cache_resource
def load_models():
    """Load Stage 1 and all Stage 2 models"""
    # Stage 1: Produce identifier
    stage1_model = tf.keras.models.load_model("models/stage1_classifier.keras")
    with open("models/stage1_classes.txt", "r") as f:
        stage1_classes = [line.strip() for line in f.readlines()]

    # Stage 2: Freshness classifiers
    stage2_models = {}
    stage2_classes = {}

    freshness_dir = "models/stage2_freshness_classifiers"
    for produce in stage1_classes:
        model_path = f"{freshness_dir}/{produce}_freshness.keras"
        classes_path = f"{freshness_dir}/{produce}_classes.txt"

        if os.path.exists(model_path):
            stage2_models[produce] = tf.keras.models.load_model(model_path)
            with open(classes_path, "r") as f:
                stage2_classes[produce] = [line.strip() for line in f.readlines()]

    return stage1_model, stage1_classes, stage2_models, stage2_classes


stage1_model, stage1_classes, stage2_models, stage2_classes = load_models()

# Custom CSS
st.markdown(
    """
<style>
.big-title {
    font-size: 72px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0;
}
.subtitle {
    font-size: 24px;
    text-align: center;
    color: #666;
    margin-top: 0;
}
.days-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin: 20px 0;
}
.days-number {
    font-size: 64px;
    font-weight: bold;
    margin: 10px 0;
}
.action-box {
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid;
}
.action-wait { background-color: #e3f2fd; border-color: #2196f3; }
.action-eat { background-color: #e8f5e9; border-color: #4caf50; }
.action-cook { background-color: #fff3e0; border-color: #ff9800; }
.action-discard { background-color: #ffebee; border-color: #f44336; }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown('<p class="big-title">🥗 FreshCheck</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Food Freshness Estimator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("How It Works")
    st.write("1. 📸 Upload a photo of your produce")
    st.write("2. 🤖 AI identifies what it is")
    st.write("3. 🔍 AI analyzes freshness")
    st.write("4. ⏰ Get days remaining + tips!")

    st.markdown("---")
    st.header("Supported Produce")
    for produce in stage1_classes:
        emoji = {"strawberry": "🍓", "banana": "🍌", "tomato": "🍅", "spinach": "🥬", "avocado": "🥑"}.get(
            produce, "🥗"
        )
        st.write(f"{emoji} {produce.title()}")

    st.markdown("---")
    st.header("💡 Did You Know?")
    st.info("1.3 billion tons of food is wasted globally each year - that's 1/3 of all food produced!")

# Main interface
col1, col2, col3 = st.columns([1, 1, 1])
produce_name = None
freshness_stage_raw = None
freshness_stage_key = None
freshness_confidence = None
stage1_pred = None
stage2_pred = None
data = None

with col1:
    st.header("📸 Upload Your Produce")
    uploaded_file = st.file_uploader(
        "Take a clear photo of your fruit or vegetable",
        type=["jpg", "jpeg", "png"],
        help="Best results: good lighting, close-up, plain background",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Photo", use_column_width=True)

with col2:
    st.header("🔍 Analysis Results")

    if uploaded_file:
        # Keep raw RGB pixels here; both models already preprocess internally.
        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # STAGE 1: Identify produce
        with st.spinner("🔍 Identifying produce..."):
            stage1_pred = stage1_model.predict(img_array, verbose=0)
            produce_idx = np.argmax(stage1_pred[0])
            produce_name = stage1_classes[produce_idx]
            produce_confidence = stage1_pred[0][produce_idx] * 100

        st.success(f"**Detected:** {produce_name.title()} ({produce_confidence:.0f}% confident)")

        # STAGE 2: Analyze freshness
        if produce_name in stage2_models:
            with st.spinner("🧪 Analyzing freshness..."):
                stage2_model = stage2_models[produce_name]
                stage2_pred = stage2_model.predict(img_array, verbose=0)
                freshness_idx = np.argmax(stage2_pred[0])
                freshness_stage_raw = stage2_classes[produce_name][freshness_idx]
                freshness_confidence = stage2_pred[0][freshness_idx] * 100
                freshness_stage_key = normalize_freshness_stage(produce_name, freshness_stage_raw)

            # Get data
            if produce_name in FRESHNESS_DATA and freshness_stage_key in FRESHNESS_DATA[produce_name]:
                data = FRESHNESS_DATA[produce_name][freshness_stage_key]

                # Days remaining - BIG DISPLAY
                days = data["days_remaining"]
                st.markdown(
                    f"""
                <div class="days-box">
                    <div style="font-size: 24px;">Estimated Shelf Life</div>
                    <div class="days-number">{days}</div>
                    <div style="font-size: 28px;">day{"s" if days != 1 else ""}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Freshness stage
                st.metric("Freshness Stage", freshness_stage_key.replace("_", " ").title(), f"{freshness_confidence:.0f}% confident")

                # Description
                st.info(f"**Visual Signs:** {data['description']}")
            else:
                st.warning(f"Predicted stage '{freshness_stage_raw}' is not mapped to the freshness database yet.")
        else:
            st.warning(f"Freshness model for {produce_name} not yet trained!")

with col3:
    st.header("🌍 Waste Impact Converter")
    st.caption("Quick estimate of impact avoided if this produce is used instead of wasted.")

    if produce_name and produce_name in IMPACT_FACTORS:
        factors = IMPACT_FACTORS[produce_name]
        units_label = "bunches" if produce_name == "spinach" else "pieces"
        quantity = st.number_input(
            f"How many {produce_name} {units_label} are at risk of being wasted?",
            min_value=1,
            max_value=500,
            value=3 if produce_name == "banana" else 2,
            step=1,
        )

        total_kg = quantity * factors["kg_per_unit"]
        co2e_kg = total_kg * factors["co2e_kg_per_kg"]
        water_l = total_kg * factors["water_l_per_kg"]
        car_km = co2e_kg / CAR_KGCO2_PER_KM
        phone_charges = co2e_kg / PHONE_CHARGE_KGCO2
        showers = water_l / SHOWER_LITERS

        st.metric("Estimated food saved", f"{total_kg:.2f} kg")
        st.metric("CO2e avoided", f"{co2e_kg:.2f} kg CO2e")
        st.metric("Water footprint avoided", f"{water_l:.0f} L")
        st.info(
            f"Equivalent to ~{car_km:.1f} km by car, ~{phone_charges:.0f} full phone charges, "
            f"or ~{showers:.1f} showers."
        )
        if data is not None:
            st.success(f"Action now: **{data['action']}**")
    else:
        st.info("Upload a produce photo to unlock the impact converter.")

# Detailed recommendations
if uploaded_file and produce_name and data is not None and freshness_stage_key is not None:
    st.markdown("---")

    # Action recommendation
    action = data["action"]
    action_class = "action-wait"
    if "eat" in action.lower() or "enjoy" in action.lower():
        action_class = "action-eat"
    elif "cook" in action.lower() or "bake" in action.lower():
        action_class = "action-cook"
    elif "discard" in action.lower():
        action_class = "action-discard"

    st.markdown(
        f"""
    <div class="action-box {action_class}">
        <h3>🎯 Recommended Action</h3>
        <h2>{action}</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Tabs for detailed info
    tab1, tab2, tab3 = st.tabs(["🍳 Recipe Ideas", "📦 Storage Tips", "📊 All Predictions"])

    with tab1:
        st.subheader(f"What to make with {freshness_stage_key.replace('_', ' ')} {produce_name}")
        recipes = data.get("recipes", [])

        if recipes:
            for recipe in recipes:
                st.write(f"• {recipe}")

        # Special banana bread callout!
        if produce_name == "banana" and freshness_stage_key == "overripe_bread":
            st.balloons()
            st.success("🍌🍞 PERFECT FOR BANANA BREAD! Don't throw these away - they make the BEST baked goods!")

    with tab2:
        st.subheader("Storage Tips")
        st.write(data["storage_tips"])

        # Show expiration date
        if days > 0:
            expiry_date = datetime.now() + timedelta(days=days)
            st.warning(f"⏰ Use by: **{expiry_date.strftime('%A, %B %d, %Y')}**")

        # Fun fact
        if produce_name in FOOD_WASTE_FACTS:
            st.info(f"💡 {FOOD_WASTE_FACTS[produce_name]}")

    with tab3:
        st.subheader("Model Confidence Breakdown")

        # Stage 1 predictions
        st.write("**Stage 1: Produce Identification**")
        for i, cls in enumerate(stage1_classes):
            conf = stage1_pred[0][i] * 100
            st.progress(conf / 100, text=f"{cls.title()}: {conf:.1f}%")

        st.write("**Stage 2: Freshness Analysis**")
        for i, cls in enumerate(stage2_classes[produce_name]):
            conf = stage2_pred[0][i] * 100
            st.progress(conf / 100, text=f"{cls.replace('_', ' ').title()}: {conf:.1f}%")

# Interactive question (optional)
if uploaded_file and "show_texture_question" not in st.session_state:
    st.session_state.show_texture_question = True

if uploaded_file and produce_name and st.session_state.show_texture_question:
    st.markdown("---")
    st.header("🤔 Help Us Improve Accuracy")

    texture = st.radio(
        f"When you touch this {produce_name}, how does it feel?",
        ["I haven't checked yet", "Hard/Firm", "Slightly soft", "Very soft/Mushy", "Slimy"],
        horizontal=True,
    )

    if texture != "I haven't checked yet":
        st.success("Thanks! Your feedback helps improve our model.")
        # Here you could adjust the prediction or log feedback
        if texture == "Slimy" and freshness_stage_key != "spoiled":
            st.warning("⚠️ Slimy texture often means spoilage - be cautious even if it looks okay!")
        if texture == "Hard/Firm" and freshness_stage_key == "fresh":
            st.warning("⚠️ If it's very firm, it might be underripe. Check back in a few days!")
        if texture == "Very soft/Mushy" and freshness_stage_key in ["fresh", "medium"]:
            st.warning("⚠️ If it's mushy, it might be overripe or spoiled. Use soon or check for mold/smell!")

# Footer
st.markdown("---")
st.markdown(
    """
**About FreshCheck**  
Two-stage hierarchical classification system using transfer learning (MobileNetV2)  
- Stage 1: Produce identification (5 types)  
- Stage 2: Freshness analysis (3-5 stages per type)  

**Environmental Impact:** Food waste generates 8% of global GHG emissions. Every item you save matters!

**Created by:** Rebeca Borrego Cavazos  
**GitHub:** github.com/rebeca-bc/freshcheck
"""
)

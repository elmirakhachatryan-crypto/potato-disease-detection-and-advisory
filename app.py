"""
Կարտոֆիլի հիվանդությունների ախտորոշիչ
Հայաստանի ֆերմերների համար
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image 
import gdown
import os
st.set_page_config(
    page_title="Կարտոֆիլի ախտորոշիչ",
    page_icon="🥔",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .healthy { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .disease { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; }
    .warning { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: white; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "potato_weights.weights.h5")
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"Մոդելը չբեռնվեց։ {e}")
        return None

DISEASE_INFO = {
    'Potato___healthy': {
        'name_hy': '✅ Առողջ կարտոֆիլ',
        'severity': 'low',
        'description': 'Շնորհավորում ենք! Ձեր կարտոֆիլը լիովին առողջ է։',
        'symptoms': [],
        'recommendations': [
            '💧 Շարունակեք կանոնավոր ջրել',
            '🌱 Օգտագործեք օրգանական պարարտանյութեր',
            '👁️ Կանոնավոր ստուգեք տերևները',
        ],
        'urgent': False
    },
    'Potato___Early_blight': {
        'name_hy': '⚠️ Վաղ մանրէ (Early Blight)',
        'severity': 'medium',
        'description': 'Վաղ մանրէ հայտնաբերված է։ Անհրաժեշտ է բուժում։',
        'recommendations': [
            '✂️ Հեռացրեք վարակված տերևները',
            '💊 Ցողեք համապատասխան դեղերով',
        ],
        'urgent': False
    },
    'Potato___Late_blight': {
        'name_hy': '🚨 Ուշ մանրէ (Late Blight)',
        'severity': 'high',
        'description': 'ՎՏԱՆԳ! Անհապաղ գործողություն է պետք!',
        'recommendations': [
            '🚨 Հեռացրեք վարակված բույսերը',
            '💊 Կիրառեք բուժում',
        ],
        'urgent': True
    }
}


def predict_disease(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    idx = np.argmax(predictions[0])
    cls = class_names[idx]
    confidence = float(predictions[0][idx] * 100)

    probs = {
        class_names[i]: float(predictions[0][i] * 100)
        for i in range(len(class_names))
    }

    return cls, confidence, probs


def main():
    st.markdown('<div class="main-header">🥔 Կարտոֆիլի հիվանդությունների ախտորոշիչ</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    image = None
    uploaded_file = None

    with col1:
        uploaded_file = st.file_uploader("Վերբեռնեք կարտոֆիլի տերևի լուսանկար", type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)

    with col2:
        if image is not None:
            model = load_model()
            if model is None:
                return

            with st.spinner("Վերլուծում ենք..."):
                prediction, confidence, all_probs = predict_disease(image, model)

            info = DISEASE_INFO[prediction]
            box_class = 'healthy' if info['severity'] == 'low' else ('warning' if info['severity'] == 'medium' else 'disease')

            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{info['name_hy']}</h2>
                <h3>Վստահություն: {confidence:.1f}%</h3>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            if info['urgent']:
                st.error("🚨 Անհապաղ գործողություն է պետք!")

            st.markdown("### 💊 Խորհուրդներ")
            for r in info['recommendations']:
                st.write(r)

            st.markdown("### 📊 Կանխատեսումներ")
            for c, p in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{c}: {p:.1f}%")

if __name__ == "__main__":
    main()

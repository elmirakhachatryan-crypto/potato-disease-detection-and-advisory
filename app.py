"""
Կարտոֆիլի հիվանդությունների ախտորոշիչ
Հայաստանի ֆերմերների համար
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('/content/potato_disease_model.keras')
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
        'symptoms': [
            '🟤 Շագանակագույն բծեր տերևների վրա',
            '⭕ Կոնցենտրիկ օղակներ',
            '🍂 Տերևների աստիճանական չորանում',
        ],
        'recommendations': [
            '✂️ Հեռացրեք վարակված տերևները',
            '💊 Ցողեք Dithane M-45 կամ Bravo պղնձի սուլֆատով',
            '🔄 Կրկնեք 10-14 օրը մեկ',
            '🌿 Պահպանեք բույսերի միջև հեռավորությունը',
        ],
        'urgent': False
    },
    'Potato___Late_blight': {
        'name_hy': '🚨 Ուշ մանրէ (Late Blight)',
        'severity': 'high',
        'description': 'ՎՏԱՆԳ! Ուշ մանրէ հայտնաբերված է։ Անհապաղ գործողություն պահանջվում է!',
        'symptoms': [
            '💧 Ջրային, մուգ բծեր տերևների վրա',
            '⚫ Արագ մգանում և փտում',
            '🌫️ Սպիտակ բորբոս տերևի հետևից',
            '🥔 Պալարների վարակ',
        ],
        'recommendations': [
            '🚨 ԱՆՄԻՋԱՊԵՍ հեռացրեք բոլոր վարակված բույսերը',
            '💊 Ցողեք Ridomil Gold Plus կամ Infinito',
            '🔄 Կրկնեք 5-7 օրը մեկ',
            '🚫 Մի ոռոգեք գիշերը',
            '📞 Կապ հաստատեք ագրոնոմի հետ',
        ],
        'urgent': True
    }
}

def predict_disease(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_names = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    all_probs = {class_names[i]: float(predictions[0][i] * 100) for i in range(len(class_names))}
    return predicted_class, confidence, all_probs

def main():
    st.markdown('<div class="main-header">🥔 Կարտոֆիլի հիվանդությունների ախտորոշիչ</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#666; margin-bottom:2rem;">🇦🇲 Հայաստանի ֆերմերների համար</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Նկար վերբեռնել")
        uploaded_file = st.file_uploader("Վերբեռնեք կարտոֆիլի տերևի լուսանկարը", type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Ձեր նկարը", use_container_width=True)

    with col2:
        if uploaded_file:
            model = load_model()
            if model is None:
                return

            with st.spinner('🔍 Վերլուծում ենք...'):
                prediction, confidence, all_probs = predict_disease(image, model)

            disease_info = DISEASE_INFO[prediction]
            severity = disease_info['severity']
            box_class = 'healthy' if severity == 'low' else ('warning' if severity == 'medium' else 'disease')

            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{disease_info['name_hy']}</h2>
                <h3>Վստահություն: {confidence:.1f}%</h3>
                <p>{disease_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            if disease_info['urgent']:
                st.error("🚨 ԱՆՀԱՊԱՂ ԳՈՐԾՈՂՈՒԹՅՈՒՆՆԵՐ ՊԱՀԱՆՋՎՈՒՄ ԵՆ!")

            if disease_info['symptoms']:
                st.markdown("### 🔬 Ախտանիշներ")
                for symptom in disease_info['symptoms']:
                    st.markdown(f"- {symptom}")

            st.markdown("### 💊 Խորհուրդներ")
            for rec in disease_info['recommendations']:
                st.markdown(f"- {rec}")

            st.markdown("### 📊 Բոլոր կանխատեսումները")
            for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                label = DISEASE_INFO[cls]['name_hy']
                st.progress(int(prob), text=f"{label}: {prob:.1f}%")

if __name__ == "__main__":
    main()

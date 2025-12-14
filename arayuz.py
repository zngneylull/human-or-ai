import streamlit as st
import joblib
import os
import pandas as pd
import time
from deep_translator import GoogleTranslator

# --- SAYFA AYARLARI (Senin Modern Ayarların) ---
st.set_page_config(
    page_title="Turing Turnusolu",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS İLE GÖRSEL İYİLEŞTİRME ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0rem !important;
        max-width: 1400px !important;
    }
    
    header {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
    .stTitleAnchor {
        display: none !important;
    }

    .stTextArea textarea {font-size: 16px !important;}
            
    div[data-testid="stMetricValue"] {font-size: 28px !important;}
    
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        transition: 0.3s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #e0e0e0 !important;
        border-color: #cccccc !important;
        color: black !important;
        font-weight: bold !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #c0c0c0 !important;
    }
    
    
    #MainMenu {display: none !important;}
    footer {display: none !important;}
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
st.markdown("""
    <h1 style='text-align: left; margin-bottom: -15px; margin-top: -50px;'>
        🧪 Turing Turnusolu
    </h1>
    <h3 style='text-align: left; color: gray; margin-bottom: 20px;'>
        Metin Kaynak Sınama Platformu
    </h3>
    <p style='font-size: 1.1rem;'>
        Bu sistem, metinlere dijital bir turnusol testi uygulayarak 
        metnin <b>İnsan</b> mı yoksa <b>Yapay Zeka</b> mı olduğunu tespit eder.
    </p>
    <hr style="margin-top: 0.5rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def dosyalari_yukle():
    if not os.path.exists('vectorizer.pkl'): return None, None
    try:
        vec = joblib.load('vectorizer.pkl')
        models = {
            '🌲 Random Forest (En İyisi)': joblib.load('model_random_forest.pkl'),
            '📈 Logistic Regression (Dengeli)': joblib.load('model_logistic.pkl'),
            '🔢 Naive Bayes (Hızlı)': joblib.load('model_naive_bayes.pkl')
        }
        return vec, models
    except: return None, None

vectorizer, models = dosyalari_yukle()
if vectorizer is None:
    st.error("🚨 HATA: Model dosyaları bulunamadı! Lütfen 'modelleri_hazirla.py' çalıştırın.")
    st.stop()

# --- SESSION STATE & TEMİZLEME ---
if 'text_input' not in st.session_state: st.session_state.text_input = ""
def temizle(): st.session_state.text_input = ""

# --- ANA ARAYÜZ ---

# 1. KONTROL PANELİ
col_input, col_controls = st.columns([3, 2], gap="medium")

with col_input:
      st.subheader("📝 Metin Girişi")
      metin = st.text_area(
            "Analiz edilecek metni buraya yapıştır:",
            key="text_input",
            height=250,
            placeholder="Abstract text goes here..."
        )
      st.caption(f"Karakter Sayısı: {len(metin)}")

with col_controls:
        st.subheader("⚙️ Test Konfigürasyonu")
        st.write("Hangi modeli kullanmak istersin?")
        
        secilen_model_ismi = st.radio(
            "Algoritma Seçimi:",
            list(models.keys()),
            horizontal=True,
            label_visibility="collapsed",
            index=None
        )
        
        if secilen_model_ismi:
            st.info(f"📌 **Seçili Algoritma:** {secilen_model_ismi}")
        else:
            st.warning("👈 **Lütfen test için bir model seçiniz.**")

        st.divider()
        
        b1, b2 = st.columns(2)
        with b1:
            st.button("🧹 **Temizle**", on_click=temizle, use_container_width=True)
        with b2:
            analiz_baslat = st.button("🔍 **TURNUSOL TESTİNİ BAŞLAT**", type="primary", use_container_width=True)

# --- SONUÇ ALANI ---
if analiz_baslat:
    st.divider()
    if len(metin) < 15:
        st.warning("⚠️ Yetersiz Veri: Lütfen analiz için daha uzun bir metin girin.")
    else:
        with st.spinner('Turing Turnusolu devrede... Analiz yapılıyor...'):
            time.sleep(0.5) 
            
            # 1. SESSİZ ÇEVİRİ KATMANI (Silent Translation)
            try:
                # Arka planda İngilizceye çevir
                ceviri = GoogleTranslator(source='auto', target='en').translate(metin)
                islenen_metin = ceviri 
            except Exception:
                # İnternet yoksa olduğu gibi kullan
                islenen_metin = metin

            # 2. SEÇİLEN MODELİN TAHMİNİ (Çevrilmiş metinle)
            model = models[secilen_model_ismi]
            vec_text = vectorizer.transform([islenen_metin]) # Çeviriyi vektöre çevir
            prediction = model.predict(vec_text)[0]
            try:
                proba = model.predict_proba(vec_text)[0]
                guven = proba[prediction] * 100
            except: guven = 0.0

        # --- TANI RAPORU KARTI (Senin Tasarımın + Yeni İsteğin) ---
        st.header("📊 Analiz Raporu ve Sonuçlar")
        
        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            if prediction == 1:
                # AI SONUCU (İstediğin net cümle burada)
                st.error("### 🤖 TESPİT: YAPAY ZEKA (AI)")
                st.markdown(f"""
                <div style='background-color:#ffe6e6; padding:10px; border-radius:5px; border-left: 5px solid #dc3545;'>
                    <h3 style='color:#dc3545; margin:0;'>%{guven:.1f} oranında YAPAY ZEKA tespit edildi.</h3>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Analiz: Metin, yapay zeka modellerine özgü istatistiksel izler taşıyor.")
            else:
                # İNSAN SONUCU (İstediğin net cümle burada)
                st.success("### 👤 TESPİT: İNSAN YAZIMI")
                st.markdown(f"""
                <div style='background-color:#e6fffa; padding:10px; border-radius:5px; border-left: 5px solid #28a745;'>
                    <h3 style='color:#28a745; margin:0;'>%{guven:.1f} oranında İNSAN tarafından yazılmıştır.</h3>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Analiz: Metin, doğal insan yazımına özgü çeşitlilik gösteriyor.")
        
        with res_col2:
            st.metric("Algoritma Güveni", f"%{guven:.1f}")
            if prediction == 1:
                 st.progress(int(guven), text="AI Olasılığı")
            else:
                 st.progress(int(guven), text="İnsan Olasılığı")

        with res_col3:
            st.write("**Test Detayları:**")
            st.caption(f"Modül: **{secilen_model_ismi.split(' (')[0]}**")
            st.caption(f"Dil İşleme: **Aktif**")
            st.caption(f"Durum: **Tamamlandı**")
            st.caption(f"Tarih: **{time.strftime('%d.%m.%Y')}**")

        # --- MODEL TUTARLILIK KONTROLÜ TABLOSU ---
        st.write("")
        st.write("---")
        st.subheader("🔍 Model Tutarlılık Kontrolü (Consistency Check)")
        st.caption("Aynı metnin farklı algoritmalar tarafından nasıl sınıflandırıldığını aşağıda görebilirsiniz.")

        sonuclar_listesi = []
        
        # Tüm modelleri döngüye sok (Çevrilmiş metni kullanırlar)
        for ad, mdl in models.items():
                pred = mdl.predict(vec_text)[0] # vec_text zaten çevrilmiş metnin vektörü
                try:
                    prob = mdl.predict_proba(vec_text)[0]
                    conf = prob[pred] * 100
                except: conf = 0.0
                
                durum = "🤖 AI (Yapay Zeka)" if pred == 1 else "👤 İnsan"
                
                sonuclar_listesi.append({
                    "Algoritma": ad,
                    "Tahmin Sonucu": durum,
                    "Güven Skoru": conf
                })
            
        df_sonuc = pd.DataFrame(sonuclar_listesi)

        # Tabloyu göster
        st.dataframe(
                df_sonuc,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Algoritma": st.column_config.TextColumn("Algoritma", width="medium"),
                    "Tahmin Sonucu": st.column_config.TextColumn("Tahmin", width="medium"),
                    "Güven Skoru": st.column_config.ProgressColumn(
                        "Emin Olma Oranı",
                        format="%d%%",
                        min_value=0,
                        max_value=100
                    )
                }
        )
# --- ALT BİLGİ ---
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: grey;'>Turing Turnusol Project | 2025</div>", unsafe_allow_html=True)
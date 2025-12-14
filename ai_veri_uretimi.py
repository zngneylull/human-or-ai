import pandas as pd
import time
import os
from huggingface_hub import InferenceClient

# --- AYARLAR ---
# Hugging Face anahtarını buraya yapıştır
HF_TOKEN = "" 

# YEDEK PLANI: Sırayla denenecek modeller.
# Biri çalışmazsa kod otomatik diğerine geçer.
MODEL_LISTESI = [
    "Qwen/Qwen2.5-7B-Instruct",       # 1. Tercih: Çok güçlü ve ücretsiz
    "microsoft/Phi-3-mini-4k-instruct", # 2. Tercih: Microsoft'un hızlı modeli
    "google/gemma-1.1-7b-it",         # 3. Tercih: Google'ın açık modeli
    "HuggingFaceH4/zephyr-7b-beta"    # 4. Tercih: Eski favori
]

client = InferenceClient(token=HF_TOKEN)

def calisan_modeli_bul():
    """Hangi modelin şu an ücretsiz sunucuda aktif olduğunu bulur."""
    print("🔍 Aktif ve ücretsiz model aranıyor...")
    
    for model_adi in MODEL_LISTESI:
        print(f"   Deneniyor: {model_adi} ... ", end="")
        try:
            # Ufak bir test isteği
            client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model=model_adi,
                max_tokens=10
            )
            print("✅ BAŞARILI! Bu model kullanılacak.")
            return model_adi
        except Exception as e:
            print(f"❌ (Hata: {str(e)[:50]}...)") # Hatanın başını göster
            continue
            
    print("\n🚨 HİÇBİR MODEL ÇALIŞMADI. API Kotanız bitmiş veya Hugging Face sunucuları dolu olabilir.")
    return None

def hf_ile_uret():
    # 1. Çalışan modeli seç
    secilen_model = calisan_modeli_bul()
    if not secilen_model:
        return # Hiçbiri çalışmazsa çık

    print(f"\n🚀 ÜRETİM BAŞLIYOR (Model: {secilen_model})")
    
    try:
        insan_df = pd.read_excel("insan_veri_seti_karisik.xlsx")
    except FileNotFoundError:
        print("HATA: 'insan_veri_seti_karisik.xlsx' dosyası bulunamadı!")
        return

    dosya_adi = "ai_veri_seti.xlsx"
    ai_veriler = []

    # Kaldığımız yeri kontrol
    if os.path.exists(dosya_adi):
        try:
            mevcut = pd.read_excel(dosya_adi)
            ai_veriler = mevcut.to_dict('records')
            print(f"♻️ Önceki dosyadan {len(ai_veriler)} veri yüklendi.")
        except:
            pass
    
    kalan_df = insan_df.iloc[len(ai_veriler):]

    for index, row in kalan_df.iterrows():
        if len(ai_veriler) >= 3000:
            print("🏁 HEDEFE ULAŞILDI!")
            break

        metin = str(row['Metin'])[:500]
        
        # Basit, net prompt
        messages = [
            {"role": "user", "content": f"Rewrite this academic abstract in English using different words. Do not explain, just output the text:\n\n{metin}"}
        ]

        basarili = False
        deneme = 0
        
        while not basarili and deneme < 3:
            try:
                response = client.chat_completion(
                    messages=messages,
                    model=secilen_model, 
                    max_tokens=300,
                    temperature=0.7
                )
                
                cevap = response.choices[0].message.content

                if cevap:
                    ai_veriler.append({
                        "Metin": cevap.strip(),
                        "Etiket": "AI",
                        "Konu": row.get('Konu', 'Genel'),
                        "Kaynak": f"HF-{secilen_model}"
                    })
                    
                    print(f"[{len(ai_veriler)}/3000] ✅ Üretildi.")
                    
                    if len(ai_veriler) % 10 == 0:
                        pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)
                    
                    basarili = True
                    time.sleep(2) 

            except Exception as e:
                deneme += 1
                hata = str(e)
                # Eğer model aniden hata verirse (503 vs), kısa bekle
                print(f"⚠️ Hata: {hata[:50]}... Tekrar deneniyor.")
                time.sleep(5)

    pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)
    print(f"\n🎉 İŞLEM BİTTİ! Dosya: {dosya_adi}")

if __name__ == "__main__":
    hf_ile_uret()
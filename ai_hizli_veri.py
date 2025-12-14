import pandas as pd
import os
import glob
from groq import Groq
import time

# --- ANAHTARINI BURAYA YAPIŞTIR ---
API_KEY = ""

client = Groq(api_key=API_KEY)

# DÜZELTME: 'instant' modeli en hızlısıdır. 70b yerine bunu kullanıyoruz.
MODEL_ID = "llama-3.1-8b-instant" 

def dosya_bul():
    dosyalar = glob.glob("*.xlsx")
    for dosya in dosyalar:
        if "insan" in dosya.lower():
            return dosya
    if os.path.exists("insan_veri_seti_karisik.xlsx"):
        return "insan_veri_seti_karisik.xlsx"
    return None

def groq_ile_bitir():
    print(f"🚀 GROQ TURBO (Model: {MODEL_ID}) BAŞLATILIYOR...")
    
    bulunan_dosya = dosya_bul()
    if not bulunan_dosya:
        print("HATA: Dosya bulunamadı.")
        return

    try:
        insan_df = pd.read_excel(bulunan_dosya)
    except:
        print("HATA: Dosya okunamadı. Excel kapalı mı?")
        return

    dosya_adi = "ai_veri_seti.xlsx"
    ai_veriler = []

    if os.path.exists(dosya_adi):
        try:
            df_mevcut = pd.read_excel(dosya_adi)
            ai_veriler = df_mevcut.to_dict('records')
            print(f"♻️ Kaldığı yerden devam: {len(ai_veriler)} veri hazır.")
        except:
            pass
    
    kalan_df = insan_df.iloc[len(ai_veriler):]
    print(f"🎯 Hedef: Kalan {3000 - len(ai_veriler)} veriyi bitirmek.\n")

    for index, row in kalan_df.iterrows():
        if len(ai_veriler) >= 3000:
            print("🏁 3000 Veri Tamamlandı!")
            break

        metin = str(row['Metin'])[:800]
        
        # Prompt
        prompt = f"Rewrite this academic abstract in English using different words. Keep it formal. Output ONLY the text, no intro:\n\n{metin}"

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_ID,
                temperature=0.7,
            )
            cevap = chat_completion.choices[0].message.content

            if cevap:
                ai_veriler.append({
                    "Metin": cevap.strip(),
                    "Etiket": "AI",
                    "Konu": row.get('Konu', 'Genel'),
                    "Kaynak": "Groq-Llama3-Instant"
                })
                print(f"[{len(ai_veriler)}/3000] ⚡ Üretildi")
                
                if len(ai_veriler) % 50 == 0:
                    pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)
                    print("💾 Kayıt yapıldı.")
                
                # Model çok hızlı olduğu için Groq bizi banlamasın diye 0.3 sn mola
                time.sleep(0.3)

        except Exception as e:
            if "429" in str(e): 
                print("⏳ Hız sınırına geldik, 10 sn mola...")
                time.sleep(10)
            else:
                print(f"⚠️ Hata: {e}")
                time.sleep(1)

    pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)
    print(f"\n🎉 BİTTİ! Dosya: {dosya_adi}")

if __name__ == "__main__":
    groq_ile_bitir()
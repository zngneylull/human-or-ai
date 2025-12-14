import pandas as pd
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import InferenceClient

# --- AYARLAR ---
HF_TOKEN = ""  # Anahtarını buraya tekrar yapıştır

# Modeller (Yedekli Sistem)
MODEL_LISTESI = [
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta"
]

client = InferenceClient(token=HF_TOKEN)
dosya_kilit = threading.Lock() # Dosyaya aynı anda yazılmasını önlemek için kilit

def calisan_modeli_bul():
    """Hızlıca çalışan bir model seçer."""
    for model in MODEL_LISTESI:
        try:
            client.chat_completion(messages=[{"role":"user","content":"hi"}], model=model, max_tokens=5)
            return model
        except:
            continue
    return MODEL_LISTESI[0] # Hiçbiri yanıt vermezse ilkini zorla

SECILEN_MODEL = calisan_modeli_bul()
print(f"🚀 HIZLANDIRILMIŞ MOD BAŞLATILIYOR! Model: {SECILEN_MODEL}")

def tek_satir_isleme(row, model_adi):
    """Tek bir satırı işleyen fonksiyon (İşçiler bunu kullanacak)"""
    metin = str(row['Metin'])[:500]
    messages = [{"role": "user", "content": f"Rewrite this academic abstract in English using different words. Output only the text:\n\n{metin}"}]
    
    for _ in range(3): # 3 kere deneme hakkı
        try:
            response = client.chat_completion(
                messages=messages,
                model=model_adi,
                max_tokens=300,
                temperature=0.7
            )
            cevap = response.choices[0].message.content
            if cevap:
                return {
                    "Metin": cevap.strip(),
                    "Etiket": "AI",
                    "Konu": row.get('Konu', 'Genel'),
                    "Kaynak": f"HF-{model_adi}"
                }
        except Exception as e:
            time.sleep(2) # Hata varsa 2 sn bekle tekrar dene
    return None

def hizli_uret():
    dosya_adi = "ai_veri_seti.xlsx"
    
    # 1. Mevcut Durumu Yükle
    if os.path.exists(dosya_adi):
        ai_df = pd.read_excel(dosya_adi)
        ai_veriler = ai_df.to_dict('records')
        print(f"♻️ Mevcut {len(ai_veriler)} veri yüklendi. Kaldığı yerden devam ediliyor...")
    else:
        ai_veriler = []

    # 2. İnsan Verisini Yükle
    insan_df = pd.read_excel("insan_veri_seti_karisik.xlsx")
    
    # Sadece yapılmamış olanları al
    kalan_df = insan_df.iloc[len(ai_veriler):]
    print(f"🎯 Kalan Hedef: {3000 - len(ai_veriler)} veri daha üretilecek.")

    # 3. PARALEL İŞLEME (5 İşçi Aynı Anda Çalışacak)
    # max_workers=5 demek, aynı anda 5 istek gönderilecek demektir.
    with ThreadPoolExecutor(max_workers=5) as executor:
        gelecek_sonuclar = []
        
        # Görevleri dağıt
        for index, row in kalan_df.iterrows():
            if len(ai_veriler) + len(gelecek_sonuclar) >= 3000:
                break
            # İşçiye görevi ver
            gelecek_sonuclar.append(executor.submit(tek_satir_isleme, row, SECILEN_MODEL))

        print("\n⚡ 5 Motorlu Üretim Başladı... Lütfen Bekleyin.\n")
        
        tamamlanan = 0
        yeni_veriler_buffer = []

        for future in as_completed(gelecek_sonuclar):
            sonuc = future.result()
            if sonuc:
                yeni_veriler_buffer.append(sonuc)
                tamamlanan += 1
                
                # Terminale spam yapmamak için her 5 tanede bir yaz
                if len(ai_veriler) + tamamlanan % 5 == 0:
                    print(f"[{len(ai_veriler) + tamamlanan}/3000] ✅ ...")

                # Her 20 veride bir dosyaya kaydet (Güvenlik)
                if len(yeni_veriler_buffer) >= 20:
                    with dosya_kilit: # Yazarken diğerleri beklesin
                        # Listeye ekle
                        ai_veriler.extend(yeni_veriler_buffer)
                        # Dosyaya bas
                        pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)
                        print(f"💾 {len(ai_veriler)} adet veri kaydedildi.")
                        yeni_veriler_buffer = [] # Buffer'ı boşalt

        # Kalan son parçaları da ekle
        if yeni_veriler_buffer:
            ai_veriler.extend(yeni_veriler_buffer)
            pd.DataFrame(ai_veriler).to_excel(dosya_adi, index=False)

    print(f"\n🎉 GEÇMİŞ OLSUN! {len(ai_veriler)} veri tamamlandı.")

if __name__ == "__main__":
    hizli_uret()
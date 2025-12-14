import pandas as pd
from sklearn.utils import shuffle
import os

def veri_birlestir_ve_temizle():
    print("--- USER STORY 2: VERİ TEMİZLEME VE BİRLEŞTİRME ---")
    
    # 1. Dosyaları Okuma
    print("1. Dosyalar okunuyor...")
    try:
        df_insan = pd.read_excel("insan_veri_seti_karisik.xlsx")
        df_ai = pd.read_excel("ai_veri_seti.xlsx")
        print(f"   ✅ İnsan Verisi Okundu: {len(df_insan)} satır")
        print(f"   ✅ AI Verisi Okundu:    {len(df_ai)} satır")
    except FileNotFoundError:
        print("❌ HATA: Dosyalar bulunamadı. İsimleri kontrol et!")
        return

    # 2. Etiketleme (Labeling)
    # İnsan = 0, AI = 1
    print("\n2. Etiketleme yapılıyor (İnsan=0, AI=1)...")
    df_insan['Etiket'] = 0
    df_ai['Etiket'] = 1

    # Sadece işimize yarayacak sütunları alınıyor 
    df_insan = df_insan[['Metin', 'Etiket']]
    df_ai = df_ai[['Metin', 'Etiket']]

    # ADIM 3: Birleştirme (Concatenation)
    print("3. İki veri seti birleştiriliyor...")
    df_final = pd.concat([df_insan, df_ai], ignore_index=True)
    print(f"   - Birleştirme sonrası toplam veri: {len(df_final)}")

    
    # ADIM 4: Veri Temizliği (Data Cleaning)
    print("\n4. Temizleme İşlemleri yapılıyor...")

    # a) Duplicate (Kopya) Kontrolü
    # Tekrar eden satırları silme işlemi
    kopya_sayisi = df_final.duplicated(subset=['Metin']).sum()
    df_final = df_final.drop_duplicates(subset=['Metin'])
    print(f"   - {kopya_sayisi} adet tekrar eden (kopya) veri silindi.")

    # b) Null (Boş) Kontrolü
    # Boş hücreler içeren satırları silme işlemi
    bos_sayisi = df_final['Metin'].isnull().sum()
    df_final = df_final.dropna(subset=['Metin'])
    print(f"   - {bos_sayisi} adet boş satır silindi.")
    
    # c) Mantıksız Veri Kontrolü
    # Çok kısa metinleri silme işlemi (örneğin 50 karakterden kısa olanlar)
    kisa_veri_sayisi = df_final[df_final['Metin'].str.len() < 50].shape[0]
    df_final = df_final[df_final['Metin'].str.len() >= 50]
    print(f"   - {kisa_veri_sayisi} adet çok kısa/bozuk veri silindi.")

    # ADIM 5: Karıştırma (Shuffling)
    # Alt alta sıralı verileri (önce hepsi insan, sonra hepsi AI) rastgele karıştırma işlemi
    print("5. Veriler karıştırılıyor (Shuffle)...")
    df_final = shuffle(df_final, random_state=42).reset_index(drop=True)

    # ADIM 6: Kayıt (Export)
    # Temizlenmiş ve birleştirilmiş veriyi CSV formatında kaydetme
    print("\n6. Final dosyası kaydediliyor...")
    df_final.to_excel("final_proje_verisi.xlsx", index=False)
    df_final.to_csv("final_proje_verisi.csv", index=False)

    print(f"\n✅ İŞLEM BAŞARIYLA TAMAMLANDI!")
    print(f"Eğitime Girecek Toplam Veri: {len(df_final)}")
    print(f"   - İnsan Sınıfı (0): {len(df_final[df_final['Etiket']==0])}")
    print(f"   - AI Sınıfı (1):    {len(df_final[df_final['Etiket']==1])}")
    print("Dosyalar hazır: 'final_proje_verisi.xlsx' ve 'final_proje_verisi.csv'")

if __name__ == "__main__":
    veri_birlestir_ve_temizle()
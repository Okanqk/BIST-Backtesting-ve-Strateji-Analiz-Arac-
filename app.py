import streamlit as st
import json
import os
from datetime import datetime
import zipfile
import difflib
import re

# ============================================================
# SAYFA YAPILANDIRMASI
# ============================================================
st.set_page_config(
    page_title="English Journey",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #C8102E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .word-chip {
        display: inline-block;
        padding: 8px 14px;
        margin: 4px;
        background: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .correct-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
    .wrong-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Ana Sayfa"
if 'ilerleme' not in st.session_state:
    st.session_state.ilerleme = {
        "tamamlanan_dersler": [],
        "cozulen_testler": [],
        "cozulen_alistirmalar": [],
        "toplam_alistirma_denemesi": 0,
        "dogru_cevap_sayisi": 0,
        "basari_puani": 0
    }
if 'test_sonuclari' not in st.session_state:
    st.session_state.test_sonuclari = {}
if 'aktif_test' not in st.session_state:
    st.session_state.aktif_test = None
if 'test_cevaplari' not in st.session_state:
    st.session_state.test_cevaplari = {}
# Kelime sıralama için: her alıştırmanın seçilmiş kelime listesi
if 'siralama_secim' not in st.session_state:
    st.session_state.siralama_secim = {}

# ============================================================
# JSON DOSYA YÖNETİMİ
# ============================================================
def json_yukle(yol):
    try:
        if os.path.exists(yol):
            with open(yol, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"❌ Yükleme hatası: {e}")
    return None

def json_kaydet(yol, veri):
    try:
        os.makedirs(os.path.dirname(yol), exist_ok=True)
        with open(yol, 'w', encoding='utf-8') as f:
            json.dump(veri, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"❌ Kaydetme hatası: {e}")
        return False

def klasor_yukle(klasor):
    veriler = []
    if os.path.exists(klasor):
        for dosya in sorted(os.listdir(klasor)):
            if dosya.endswith('.json'):
                v = json_yukle(os.path.join(klasor, dosya))
                if v:
                    veriler.append(v)
    return list(reversed(veriler))

def tum_dersleri_yukle():
    return klasor_yukle("data/dersler")

def tum_testleri_yukle():
    return klasor_yukle("data/testler")

def tum_alistirmalari_yukle():
    return klasor_yukle("data/alistirmalar")

def ilerleme_kaydet():
    json_kaydet("data/ilerleme.json", st.session_state.ilerleme)

def ilerleme_yukle():
    v = json_yukle("data/ilerleme.json")
    if v:
        # Eski formatla geriye uyum
        for key in ["tamamlanan_dersler", "cozulen_testler", "cozulen_alistirmalar"]:
            if key not in v:
                v[key] = []
        for key in ["toplam_alistirma_denemesi", "dogru_cevap_sayisi", "basari_puani"]:
            if key not in v:
                v[key] = 0
        st.session_state.ilerleme = v

def zip_yedek_olustur():
    try:
        zip_dosya = f"english_journey_yedek_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_dosya, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists('data'):
                for klasor, _, dosyalar in os.walk('data'):
                    for d in dosyalar:
                        yol = os.path.join(klasor, d)
                        zipf.write(yol, os.path.relpath(yol, '.'))
        return zip_dosya
    except Exception as e:
        st.error(f"❌ ZIP hatası: {e}")
        return None

def zip_yedek_geri_yukle(zip_dosya):
    try:
        with zipfile.ZipFile(zip_dosya, 'r') as zipf:
            zipf.extractall('.')
        return True
    except Exception as e:
        st.error(f"❌ ZIP geri yükleme hatası: {e}")
        return False

# ============================================================
# YARDIMCI: CEVAP KARŞILAŞTIRMA (fuzzy)
# ============================================================
def metin_normalize(s):
    """Karşılaştırma için metni temizler: küçük harf, noktalama yok, fazla boşluk yok"""
    s = s.lower().strip()
    s = re.sub(r'[.,!?;:"\'\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def cevap_dogru_mu(kullanici_cevap, dogru_cevap, alternatifler=None, esnek=True):
    """Kullanıcı cevabını doğru cevap ve alternatiflerle karşılaştırır"""
    if not kullanici_cevap:
        return False, 0.0
    
    k = metin_normalize(kullanici_cevap)
    adaylar = [dogru_cevap] + (alternatifler or [])
    
    en_yuksek = 0.0
    for aday in adaylar:
        a = metin_normalize(aday)
        if k == a:
            return True, 1.0
        if esnek:
            oran = difflib.SequenceMatcher(None, k, a).ratio()
            en_yuksek = max(en_yuksek, oran)
    
    # %92 üzeri benzerliği doğru say (küçük yazım hataları için)
    if esnek and en_yuksek >= 0.92:
        return True, en_yuksek
    return False, en_yuksek

# ============================================================
# EGZERSİZ WIDGET'LARI
# ============================================================

def widget_kelime_siralama(alistirma, key_prefix):
    """Kelime sıralama widget'ı: kullanıcı kelimeleri tıklayarak sıralar"""
    st.write(f"**Görev:** {alistirma.get('soru', '')}")
    
    if 'aciklama' in alistirma:
        st.caption(alistirma['aciklama'])
    
    kelimeler = alistirma.get('kelimeler', [])
    dogru_siralama = alistirma.get('dogru_siralama', [])
    
    # Kullanıcının seçtiği sırayı session state'te tut
    if key_prefix not in st.session_state.siralama_secim:
        st.session_state.siralama_secim[key_prefix] = []
    
    secilen = st.session_state.siralama_secim[key_prefix]
    
    # Mevcut sırayı göster
    st.write("**Senin sıralaman:**")
    if secilen:
        sira_metni = " → ".join([f"`{k}`" for k in secilen])
        st.markdown(sira_metni)
    else:
        st.info("Aşağıdaki kelimelere tıklayarak sıraya diz")
    
    # Tıklanabilir kelime butonları
    st.write("**Kelimeler:**")
    cols = st.columns(min(len(kelimeler), 4))
    for i, kelime in enumerate(kelimeler):
        with cols[i % 4]:
            if kelime not in secilen:
                if st.button(f"➕ {kelime}", key=f"{key_prefix}_kel_{i}", use_container_width=True):
                    st.session_state.siralama_secim[key_prefix].append(kelime)
                    st.rerun()
            else:
                st.button(f"✓ {kelime}", key=f"{key_prefix}_kel_{i}_done", disabled=True, use_container_width=True)
    
    # Geri al / sıfırla
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("↩️ Geri Al", key=f"{key_prefix}_geri"):
            if secilen:
                st.session_state.siralama_secim[key_prefix].pop()
                st.rerun()
    with col2:
        if st.button("🗑️ Sıfırla", key=f"{key_prefix}_sifirla"):
            st.session_state.siralama_secim[key_prefix] = []
            st.rerun()
    with col3:
        if st.button("✅ Kontrol Et", key=f"{key_prefix}_kontrol", type="primary"):
            st.session_state.ilerleme['toplam_alistirma_denemesi'] += 1
            if secilen == dogru_siralama:
                st.session_state.ilerleme['dogru_cevap_sayisi'] += 1
                st.session_state.ilerleme['basari_puani'] += 5
                ilerleme_kaydet()
                st.markdown(f"<div class='correct-box'>🎉 <b>Doğru!</b> {' '.join(secilen)}</div>", unsafe_allow_html=True)
                if 'aciklama_sonra' in alistirma:
                    st.info(f"💡 {alistirma['aciklama_sonra']}")
            else:
                ilerleme_kaydet()
                st.markdown(f"<div class='wrong-box'>❌ <b>Yanlış.</b><br>Senin: {' '.join(secilen) if secilen else '(boş)'}<br>Doğru: <b>{' '.join(dogru_siralama)}</b></div>", unsafe_allow_html=True)
                if 'aciklama_sonra' in alistirma:
                    st.info(f"💡 {alistirma['aciklama_sonra']}")


def widget_bosluk_doldurma(alistirma, key_prefix):
    """Boşluk doldurma widget'ı"""
    st.write(f"**Görev:** {alistirma.get('soru', '')}")
    if 'aciklama' in alistirma:
        st.caption(alistirma['aciklama'])
    
    cumle = alistirma.get('cumle', '')
    st.markdown(f"### {cumle}")
    
    bosluklar = alistirma.get('bosluklar', [])  # [{ipucu, dogru_cevap, alternatifler}]
    
    cevaplar = []
    for i, bosluk in enumerate(bosluklar):
        ipucu = bosluk.get('ipucu', f'Boşluk {i+1}')
        cevap = st.text_input(f"🔹 {ipucu}", key=f"{key_prefix}_bosluk_{i}")
        cevaplar.append(cevap)
    
    if st.button("✅ Kontrol Et", key=f"{key_prefix}_bosluk_kontrol", type="primary"):
        st.session_state.ilerleme['toplam_alistirma_denemesi'] += 1
        tum_dogru = True
        sonuclar = []
        for i, bosluk in enumerate(bosluklar):
            dogru, oran = cevap_dogru_mu(
                cevaplar[i],
                bosluk.get('dogru_cevap', ''),
                bosluk.get('alternatifler', [])
            )
            sonuclar.append((dogru, cevaplar[i], bosluk.get('dogru_cevap', '')))
            if not dogru:
                tum_dogru = False
        
        if tum_dogru:
            st.session_state.ilerleme['dogru_cevap_sayisi'] += 1
            st.session_state.ilerleme['basari_puani'] += 5
            ilerleme_kaydet()
            st.markdown("<div class='correct-box'>🎉 <b>Hepsi doğru!</b></div>", unsafe_allow_html=True)
        else:
            ilerleme_kaydet()
            satirlar = []
            for dogru, k_cevap, d_cevap in sonuclar:
                isaret = "✅" if dogru else "❌"
                satirlar.append(f"{isaret} Senin: <code>{k_cevap or '(boş)'}</code> → Doğru: <b>{d_cevap}</b>")
            st.markdown(f"<div class='wrong-box'>{'<br>'.join(satirlar)}</div>", unsafe_allow_html=True)
        
        if 'aciklama_sonra' in alistirma:
            st.info(f"💡 {alistirma['aciklama_sonra']}")


def widget_serbest_ceviri(alistirma, key_prefix):
    """Serbest çeviri widget'ı: TR cümle → EN yazma"""
    yon = alistirma.get('yon', 'tr_to_en')
    soru = alistirma.get('soru', '')
    
    if yon == 'tr_to_en':
        st.write("**🇹🇷 → 🇬🇧 İngilizce'ye çevir:**")
    else:
        st.write("**🇬🇧 → 🇹🇷 Türkçe'ye çevir:**")
    
    st.markdown(f"### {soru}")
    
    if 'ipucu' in alistirma:
        with st.expander("💡 İpucu"):
            st.info(alistirma['ipucu'])
    
    cevap = st.text_area("Çevirin:", key=f"{key_prefix}_ceviri", height=80)
    
    if st.button("✅ Kontrol Et", key=f"{key_prefix}_ceviri_kontrol", type="primary"):
        st.session_state.ilerleme['toplam_alistirma_denemesi'] += 1
        dogru, oran = cevap_dogru_mu(
            cevap,
            alistirma.get('dogru_cevap', ''),
            alistirma.get('alternatifler', [])
        )
        
        if dogru:
            st.session_state.ilerleme['dogru_cevap_sayisi'] += 1
            st.session_state.ilerleme['basari_puani'] += 8
            ilerleme_kaydet()
            mesaj = f"🎉 <b>Doğru!</b>"
            if oran < 1.0:
                mesaj += f" (küçük farklar var ama kabul ettim — %{int(oran*100)} eşleşme)"
            st.markdown(f"<div class='correct-box'>{mesaj}<br>Doğru cevap: <b>{alistirma.get('dogru_cevap', '')}</b></div>", unsafe_allow_html=True)
        else:
            ilerleme_kaydet()
            st.markdown(f"<div class='wrong-box'>❌ <b>Yanlış.</b><br>Senin: <code>{cevap or '(boş)'}</code><br>Doğru: <b>{alistirma.get('dogru_cevap', '')}</b><br><small>Eşleşme: %{int(oran*100)}</small></div>", unsafe_allow_html=True)
        
        if alistirma.get('alternatifler'):
            st.caption(f"Kabul edilen diğer cevaplar: {', '.join(alistirma['alternatifler'])}")
        if 'aciklama_sonra' in alistirma:
            st.info(f"💡 {alistirma['aciklama_sonra']}")


def widget_coktan_secmeli(alistirma, key_prefix):
    """Tek soru çoktan seçmeli (kavram testi için)"""
    st.write(f"**Soru:** {alistirma.get('soru', '')}")
    secenekler = alistirma.get('secenekler', [])
    cevap = st.radio("Cevabınız:", secenekler, key=f"{key_prefix}_cs_radio", index=None)
    
    if st.button("✅ Kontrol Et", key=f"{key_prefix}_cs_kontrol", type="primary"):
        if not cevap:
            st.warning("Lütfen bir seçenek seçin")
            return
        st.session_state.ilerleme['toplam_alistirma_denemesi'] += 1
        dogru_harf = alistirma.get('cevap', '')
        if cevap[0] == dogru_harf:
            st.session_state.ilerleme['dogru_cevap_sayisi'] += 1
            st.session_state.ilerleme['basari_puani'] += 3
            ilerleme_kaydet()
            st.markdown(f"<div class='correct-box'>🎉 <b>Doğru!</b></div>", unsafe_allow_html=True)
        else:
            ilerleme_kaydet()
            st.markdown(f"<div class='wrong-box'>❌ <b>Yanlış.</b> Doğru cevap: <b>{dogru_harf}</b></div>", unsafe_allow_html=True)
        if 'aciklama_sonra' in alistirma:
            st.info(f"💡 {alistirma['aciklama_sonra']}")


def alistirma_render(alistirma, key_prefix):
    """Alıştırma tipine göre uygun widget'ı çalıştırır"""
    tip = alistirma.get('tip', 'serbest_ceviri')
    
    if tip == 'kelime_siralama':
        widget_kelime_siralama(alistirma, key_prefix)
    elif tip == 'bosluk_doldurma':
        widget_bosluk_doldurma(alistirma, key_prefix)
    elif tip == 'serbest_ceviri':
        widget_serbest_ceviri(alistirma, key_prefix)
    elif tip == 'coktan_secmeli':
        widget_coktan_secmeli(alistirma, key_prefix)
    else:
        st.warning(f"⚠️ Bilinmeyen alıştırma tipi: {tip}")

# ============================================================
# SAYFA: DERSLER
# ============================================================
def dersler_sayfasi():
    st.markdown("<h1 class='main-header'>📖 İngilizce Dersleri</h1>", unsafe_allow_html=True)
    st.write("Adım adım İngilizce öğren. Video, gramer, kelime ve alıştırmalar bir arada.")
    
    dersler_listesi = tum_dersleri_yukle()
    
    if not dersler_listesi:
        st.warning("📂 Henüz ders yüklenmemiş.")
        st.info("**Ders eklemek için:** Ayarlar → Dosya Yönetimi'nden JSON yükle")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Toplam Ders", len(dersler_listesi))
    with col2:
        tamamlanan = len(st.session_state.ilerleme['tamamlanan_dersler'])
        st.metric("✅ Tamamlanan", tamamlanan)
    with col3:
        if dersler_listesi:
            yuzde = int((tamamlanan / len(dersler_listesi)) * 100)
            st.metric("📊 İlerleme", f"%{yuzde}")
    
    st.markdown("---")
    
    for ders in dersler_listesi:
        konu_id = ders.get('konu_id', 0)
        baslik = ders.get('konu_baslik', 'İsimsiz')
        seviye = ders.get('seviye', 'başlangıç')
        tamamlandi = konu_id in st.session_state.ilerleme['tamamlanan_dersler']
        icon = "✅" if tamamlandi else "○"
        
        st.markdown(f"## {icon} {baslik}")
        st.caption(f"Seviye: {seviye.title()}  •  Konu ID: {konu_id}")
        
        with st.expander("📖 Dersi Aç", expanded=False):
            if 'aciklama' in ders:
                st.write(f"**📝** {ders['aciklama']}")
            
            if 'video_link' in ders:
                st.markdown(f"🎥 **Video:** [{ders.get('video_suresi', 'İzle')}]({ders['video_link']})")
            
            ders_icerik = ders.get('ders_icerik', {})
            if ders_icerik:
                if 'detayli_aciklama' in ders_icerik:
                    st.markdown("### 📚 Gramer / Açıklama")
                    st.write(ders_icerik['detayli_aciklama'])
                
                if 'ana_kavramlar' in ders_icerik:
                    st.markdown("### 🔑 Ana Kavramlar")
                    for k in ders_icerik['ana_kavramlar']:
                        st.write(f"• {k}")
                
                if 'tr_en_farki' in ders_icerik:
                    st.markdown("### 🔁 TR ↔ EN Mantık Farkı")
                    st.warning(ders_icerik['tr_en_farki'])
            
            # KELİME LİSTESİ
            kelimeler = ders.get('kelime_listesi', [])
            if kelimeler:
                st.markdown("### 📔 Kelime Listesi")
                for kel in kelimeler:
                    en = kel.get('en', '')
                    tr = kel.get('tr', '')
                    tur = kel.get('tur', '')
                    o_en = kel.get('ornek_en', '')
                    o_tr = kel.get('ornek_tr', '')
                    
                    with st.container():
                        st.markdown(f"**🇬🇧 {en}** _({tur})_ → 🇹🇷 {tr}")
                        if o_en:
                            st.caption(f"📌 {o_en}  →  {o_tr}")
            
            # ÖRNEK CÜMLELER
            ornekler = ders.get('ornek_cumleler', [])
            if ornekler:
                st.markdown("### 💬 Örnek Cümleler")
                for o in ornekler:
                    st.markdown(f"🇬🇧 **{o.get('en', '')}**")
                    st.caption(f"🇹🇷 {o.get('tr', '')}")
                    if 'vurgu' in o:
                        st.info(f"⚡ {o['vurgu']}")
                    st.markdown("")
            
            # ALIŞTIRMALAR (ders içi)
            alistirmalar = ders.get('alistirmalar', [])
            if alistirmalar:
                st.markdown("### 🎯 Alıştırmalar")
                for idx, alis in enumerate(alistirmalar):
                    st.markdown(f"#### Alıştırma {idx+1}")
                    alistirma_render(alis, f"ders_{konu_id}_alis_{idx}")
                    st.markdown("---")
            
            # HATIRLATMALAR
            hatirlatmalar = ders.get('hatirlatmalar', [])
            if hatirlatmalar:
                st.markdown("### 📌 Hatırlatmalar")
                for h in hatirlatmalar:
                    st.success(f"• {h}")
            
            st.markdown("---")
            
            if tamamlandi:
                st.success("✅ Bu dersi tamamladın!")
            else:
                if st.button("✓ Dersi Tamamla", key=f"ders_tamam_{konu_id}", type="primary"):
                    st.session_state.ilerleme['tamamlanan_dersler'].append(konu_id)
                    st.session_state.ilerleme['basari_puani'] += 10
                    ilerleme_kaydet()
                    st.balloons()
                    st.rerun()

# ============================================================
# SAYFA: TESTLER (çoktan seçmeli setleri)
# ============================================================
def testler_sayfasi():
    st.markdown("<h1 class='main-header'>🎯 Testler</h1>", unsafe_allow_html=True)
    
    testler = tum_testleri_yukle()
    if not testler:
        st.warning("📂 Henüz test yüklenmemiş.")
        st.info("Test JSON'larını `data/testler/` klasörüne ekle (Ayarlar sayfasından).")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Toplam Test", len(testler))
    with col2:
        st.metric("✅ Çözülen", len(st.session_state.ilerleme['cozulen_testler']))
    with col3:
        st.metric("⭐ Başarı Puanı", st.session_state.ilerleme.get('basari_puani', 0))
    
    st.markdown("---")
    
    for test_data in testler:
        test_id = test_data.get('konu_id', 0)
        konu = test_data.get('konu_baslik', 'İsimsiz Test')
        sorular = test_data.get('test_sorulari', [])
        
        if not sorular:
            continue
        
        cozuldu = test_id in st.session_state.ilerleme['cozulen_testler']
        icon = "✅" if cozuldu else "○"
        
        with st.expander(f"{icon} {konu} — {len(sorular)} soru", expanded=False):
            if cozuldu:
                skor = st.session_state.test_sonuclari.get(test_id, 0)
                st.success(f"✅ Çözüldü! Skor: {skor}/{len(sorular)}")
                if st.button(f"🔄 Tekrar Çöz", key=f"tekrar_{test_id}"):
                    st.session_state.aktif_test = test_id
                    st.session_state.test_cevaplari = {}
                    st.rerun()
            else:
                if st.button(f"▶️ Teste Başla", key=f"baslat_{test_id}", type="primary"):
                    st.session_state.aktif_test = test_id
                    st.session_state.test_cevaplari = {}
                    st.rerun()
            
            if st.session_state.aktif_test == test_id:
                st.subheader("🎯 Sorular")
                for idx, soru in enumerate(sorular):
                    st.write(f"**Soru {idx+1}:** {soru.get('soru', '')}")
                    secenekler = soru.get('secenekler', [])
                    secim = st.radio(
                        "Cevap:",
                        secenekler,
                        key=f"test_{test_id}_s_{idx}",
                        index=None
                    )
                    if secim:
                        st.session_state.test_cevaplari[idx] = secim[0]
                    st.markdown("---")
                
                if len(st.session_state.test_cevaplari) == len(sorular):
                    if st.button("📤 Testi Gönder", type="primary", key=f"gonder_{test_id}"):
                        dogru = 0
                        for idx, soru in enumerate(sorular):
                            if st.session_state.test_cevaplari.get(idx) == soru.get('cevap'):
                                dogru += 1
                        st.session_state.test_sonuclari[test_id] = dogru
                        if test_id not in st.session_state.ilerleme['cozulen_testler']:
                            st.session_state.ilerleme['cozulen_testler'].append(test_id)
                            st.session_state.ilerleme['basari_puani'] += dogru * 5
                        ilerleme_kaydet()
                        st.session_state.aktif_test = None
                        st.session_state.test_cevaplari = {}
                        st.success(f"🎉 Tamamlandı! {dogru}/{len(sorular)} doğru")
                        st.rerun()

# ============================================================
# SAYFA: ALIŞTIRMALAR (bağımsız set)
# ============================================================
def alistirmalar_sayfasi():
    st.markdown("<h1 class='main-header'>🧩 Alıştırmalar</h1>", unsafe_allow_html=True)
    st.write("Karışık alıştırma setleri — kelime sıralama, boşluk doldurma, çeviri.")
    
    setler = tum_alistirmalari_yukle()
    if not setler:
        st.warning("📂 Henüz alıştırma seti yüklenmemiş.")
        st.info("Alıştırma JSON'larını `data/alistirmalar/` klasörüne ekle.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧩 Toplam Set", len(setler))
    with col2:
        st.metric("✅ Tamamlanan", len(st.session_state.ilerleme['cozulen_alistirmalar']))
    with col3:
        denemeler = st.session_state.ilerleme.get('toplam_alistirma_denemesi', 0)
        dogrular = st.session_state.ilerleme.get('dogru_cevap_sayisi', 0)
        if denemeler > 0:
            basari_yuzde = int((dogrular / denemeler) * 100)
            st.metric("🎯 Başarı Oranı", f"%{basari_yuzde}")
        else:
            st.metric("🎯 Başarı Oranı", "—")
    
    st.markdown("---")
    
    for set_data in setler:
        set_id = set_data.get('konu_id', 0)
        baslik = set_data.get('konu_baslik', 'İsimsiz')
        alistirmalar = set_data.get('alistirmalar', [])
        
        if not alistirmalar:
            continue
        
        tamamlandi = set_id in st.session_state.ilerleme['cozulen_alistirmalar']
        icon = "✅" if tamamlandi else "○"
        
        with st.expander(f"{icon} {baslik} — {len(alistirmalar)} alıştırma", expanded=False):
            for idx, alis in enumerate(alistirmalar):
                st.markdown(f"### Alıştırma {idx+1}")
                tip = alis.get('tip', 'serbest_ceviri')
                tip_isim = {
                    'kelime_siralama': '🔤 Kelime Sıralama',
                    'bosluk_doldurma': '🔲 Boşluk Doldurma',
                    'serbest_ceviri': '✍️ Çeviri',
                    'coktan_secmeli': '☑️ Çoktan Seçmeli'
                }.get(tip, tip)
                st.caption(f"Tip: {tip_isim}")
                
                alistirma_render(alis, f"set_{set_id}_alis_{idx}")
                st.markdown("---")
            
            if not tamamlandi:
                if st.button(f"✓ Seti Tamamla", key=f"set_tamam_{set_id}", type="primary"):
                    st.session_state.ilerleme['cozulen_alistirmalar'].append(set_id)
                    st.session_state.ilerleme['basari_puani'] += 15
                    ilerleme_kaydet()
                    st.balloons()
                    st.rerun()

# ============================================================
# SAYFA: İLERLEME
# ============================================================
def ilerleme_sayfasi():
    st.markdown("<h1 class='main-header'>📊 İlerlemen</h1>", unsafe_allow_html=True)
    
    ilerleme_yukle()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Tamamlanan Ders", len(st.session_state.ilerleme['tamamlanan_dersler']))
    with col2:
        st.metric("🎯 Çözülen Test", len(st.session_state.ilerleme['cozulen_testler']))
    with col3:
        st.metric("🧩 Tamamlanan Set", len(st.session_state.ilerleme['cozulen_alistirmalar']))
    with col4:
        st.metric("✍️ Toplam Deneme", st.session_state.ilerleme.get('toplam_alistirma_denemesi', 0))
    
    st.markdown("---")
    
    # Başarı oranı
    denemeler = st.session_state.ilerleme.get('toplam_alistirma_denemesi', 0)
    dogrular = st.session_state.ilerleme.get('dogru_cevap_sayisi', 0)
    if denemeler > 0:
        oran = dogrular / denemeler
        st.subheader(f"🎯 Başarı Oranı: %{int(oran*100)}")
        st.progress(oran)
        st.caption(f"{dogrular} doğru / {denemeler} deneme")
    
    st.markdown("---")
    
    # Başarı puanı ve seviye
    st.subheader("⭐ Başarı Puanı")
    puan = st.session_state.ilerleme.get('basari_puani', 0)
    max_puan = 1000
    st.progress(min(puan / max_puan, 1.0))
    st.write(f"**{puan} / {max_puan}** puan")
    
    if puan < 100:
        seviye = "🥉 Beginner (Başlangıç)"
    elif puan < 300:
        seviye = "🥈 Elementary (Temel)"
    elif puan < 600:
        seviye = "🥇 Intermediate (Orta)"
    elif puan < 900:
        seviye = "💎 Upper-Intermediate (İleri)"
    else:
        seviye = "🏆 Advanced (Uzman)"
    st.info(f"**Mevcut Seviye:** {seviye}")
    
    st.markdown("---")
    
    with st.expander("⚠️ Tehlikeli Alan"):
        st.warning("Tüm ilerleme verilerini sıfırla")
        if st.button("🗑️ İlerlememi Sıfırla", type="secondary"):
            st.session_state.ilerleme = {
                "tamamlanan_dersler": [],
                "cozulen_testler": [],
                "cozulen_alistirmalar": [],
                "toplam_alistirma_denemesi": 0,
                "dogru_cevap_sayisi": 0,
                "basari_puani": 0
            }
            st.session_state.test_sonuclari = {}
            ilerleme_kaydet()
            st.success("✅ Sıfırlandı")
            st.rerun()

# ============================================================
# SAYFA: AYARLAR
# ============================================================
def ayarlar_sayfasi():
    st.markdown("<h1 class='main-header'>⚙️ Ayarlar ve Veri Yönetimi</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dosya Yönetimi", "📋 Metin ile Yükleme", "💾 Yedekleme", "ℹ️ Hakkında"])
    
    # ----- TAB 1: Dosya yükleme -----
    with tab1:
        st.subheader("📥 JSON Dosya Yükleme")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**📚 Dersler**")
            f = st.file_uploader("Ders JSON", type=['json'], key="ders_upload")
            if f is not None:
                try:
                    veri = json.load(f)
                    if json_kaydet(f"data/dersler/{f.name}", veri):
                        st.success(f"✅ {f.name} yüklendi!")
                except Exception as e:
                    st.error(f"❌ {e}")
        
        with col2:
            st.info("**🎯 Testler**")
            f = st.file_uploader("Test JSON", type=['json'], key="test_upload")
            if f is not None:
                try:
                    veri = json.load(f)
                    if json_kaydet(f"data/testler/{f.name}", veri):
                        st.success(f"✅ {f.name} yüklendi!")
                except Exception as e:
                    st.error(f"❌ {e}")
        
        with col3:
            st.info("**🧩 Alıştırmalar**")
            f = st.file_uploader("Alıştırma JSON", type=['json'], key="alis_upload")
            if f is not None:
                try:
                    veri = json.load(f)
                    if json_kaydet(f"data/alistirmalar/{f.name}", veri):
                        st.success(f"✅ {f.name} yüklendi!")
                except Exception as e:
                    st.error(f"❌ {e}")
        
        st.markdown("---")
        st.subheader("📂 Yüklü Dosyalar")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**📚 Dersler:**")
            for d in tum_dersleri_yukle():
                st.write(f"• {d.get('konu_baslik', '?')}")
        with col2:
            st.write("**🎯 Testler:**")
            for t in tum_testleri_yukle():
                st.write(f"• {t.get('konu_baslik', '?')}")
        with col3:
            st.write("**🧩 Alıştırmalar:**")
            for a in tum_alistirmalari_yukle():
                st.write(f"• {a.get('konu_baslik', '?')}")
    
    # ----- TAB 2: Metin yapıştırma -----
    with tab2:
        st.subheader("📋 JSON Metin ile Yükleme")
        secim = st.radio("Tip:", ["📚 Ders", "🎯 Test", "🧩 Alıştırma"], horizontal=True)
        metin = st.text_area("JSON yapıştır:", height=400, placeholder='{"konu_id": 1, ...}')
        
        if st.button("✅ Yükle", type="primary"):
            if metin.strip():
                try:
                    veri = json.loads(metin)
                    konu_id = veri.get('konu_id', 'yeni')
                    if secim == "📚 Ders":
                        yol = f"data/dersler/ders_{konu_id}.json"
                    elif secim == "🎯 Test":
                        yol = f"data/testler/test_{konu_id}.json"
                    else:
                        yol = f"data/alistirmalar/alistirma_{konu_id}.json"
                    
                    if json_kaydet(yol, veri):
                        st.success(f"✅ Yüklendi: {yol}")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON hatası: {e}")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("Lütfen JSON yapıştır")
        
        st.markdown("---")
        st.subheader("📝 Şema Örnekleri")
        
        if st.checkbox("📚 Ders şeması göster"):
            st.code(ORNEK_DERS_JSON, language='json')
        
        if st.checkbox("🎯 Test şeması göster"):
            st.code(ORNEK_TEST_JSON, language='json')
        
        if st.checkbox("🧩 Alıştırma şeması göster"):
            st.code(ORNEK_ALISTIRMA_JSON, language='json')
        
        if st.checkbox("🤖 YZ Prompt Template göster (önemli!)"):
            st.code(YZ_PROMPT_TEMPLATE, language='text')
    
    # ----- TAB 3: Yedekleme -----
    with tab3:
        st.subheader("💾 Yedekleme")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Yedek Oluştur", type="primary", use_container_width=True):
                zd = zip_yedek_olustur()
                if zd:
                    with open(zd, 'rb') as f:
                        st.download_button(
                            "📥 Yedek İndir",
                            data=f,
                            file_name=zd,
                            mime="application/zip"
                        )
                    st.success(f"✅ Oluşturuldu: {zd}")
        
        with col2:
            st.write("**📤 Geri Yükle**")
            yzip = st.file_uploader("ZIP", type=['zip'], key="yedek_upload")
            if yzip is not None:
                if st.button("🔄 Geri Yükle", use_container_width=True):
                    try:
                        with open("temp_yedek.zip", "wb") as f:
                            f.write(yzip.getbuffer())
                        if zip_yedek_geri_yukle("temp_yedek.zip"):
                            st.success("✅ Geri yüklendi!")
                            os.remove("temp_yedek.zip")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
    
    # ----- TAB 4: Hakkında -----
    with tab4:
        st.subheader("ℹ️ English Journey")
        st.markdown("""
        ### 🇬🇧 English Journey v1.0
        
        **Özellikler:**
        - 📖 İnteraktif İngilizce dersleri (video + gramer + kelime + alıştırma)
        - 🎯 Çoktan seçmeli testler
        - 🧩 4 farklı alıştırma tipi:
          - 🔤 Kelime sıralama (TR mantığında dizip EN'e geçiş için)
          - 🔲 Boşluk doldurma (gramer kuralları için)
          - ✍️ Serbest çeviri (üretme becerisi için, fuzzy match'li)
          - ☑️ Çoktan seçmeli (kavram testi için)
        - 📊 Detaylı ilerleme takibi (başarı oranı + seviye sistemi)
        - 📁 JSON tabanlı içerik (YZ ile kolay üretim)
        - 💾 ZIP yedekleme
        
        **JSON şemaları:** Metin ile Yükleme sekmesinde örnekler var.
        **YZ Prompt Template:** Aynı sekmede, Gemini/ChatGPT'ye verip ders üretebilirsin.
        """)

# ============================================================
# ŞEMA ÖRNEKLERİ (Ayarlar sayfasında gösterilir)
# ============================================================
ORNEK_DERS_JSON = '''{
  "konu_id": 1,
  "konu_baslik": "Çoğul -s Kuralı ve Geniş Zaman",
  "aciklama": "İngilizce'de çoğul ekinin zorunlu kullanımı ve TR ile farkları",
  "seviye": "başlangıç",
  "video_link": "https://www.youtube.com/watch?v=...",
  "video_suresi": "42:15",
  "ders_icerik": {
    "detayli_aciklama": "Hocanın anlattığı kuralın açık metni...",
    "ana_kavramlar": [
      "Çoğul -s eki zorunluluğu",
      "Geniş zamanda 3. tekil şahıs -s"
    ],
    "tr_en_farki": "TR'de 'Yazarlar kitap yazar' yeterliyken, EN'de 'Writers write books' — kitap da çoğul olmalı."
  },
  "kelime_listesi": [
    {
      "en": "writer",
      "tr": "yazar",
      "tur": "isim",
      "ornek_en": "Writers write books.",
      "ornek_tr": "Yazarlar kitap yazar."
    }
  ],
  "ornek_cumleler": [
    {
      "en": "Teachers teach students.",
      "tr": "Öğretmenler öğrenci öğretir.",
      "vurgu": "Hem teachers hem students çoğul."
    }
  ],
  "alistirmalar": [
    {
      "tip": "kelime_siralama",
      "soru": "Kelimeleri doğru sırada diz: 'Yazarlar kitap yazar'",
      "aciklama": "Önce TR mantığında: özne + fiil + nesne",
      "kelimeler": ["books", "write", "writers"],
      "dogru_siralama": ["writers", "write", "books"],
      "aciklama_sonra": "Hem yazarlar hem kitaplar çoğul olduğu için her ikisinde de -s var."
    },
    {
      "tip": "bosluk_doldurma",
      "soru": "Boşlukları doldur",
      "cumle": "Doctors ___ patients in the hospital.",
      "bosluklar": [
        {
          "ipucu": "fiil (treat)",
          "dogru_cevap": "treat",
          "alternatifler": []
        }
      ],
      "aciklama_sonra": "Çoğul özne ile fiil yalın hali alır."
    },
    {
      "tip": "serbest_ceviri",
      "yon": "tr_to_en",
      "soru": "Doktorlar hasta tedavi eder.",
      "ipucu": "Hem doktorlar hem hastalar çoğul olmalı.",
      "dogru_cevap": "Doctors treat patients.",
      "alternatifler": ["The doctors treat patients", "Doctors treat the patients"],
      "aciklama_sonra": "TR'de 'hasta' tek söylenir, EN'de 'patients' çoğul."
    }
  ],
  "hatirlatmalar": [
    "İngilizce'de genel doğruluk ifadelerinde çoğul kullan.",
    "TR'de tek çoğul yeterli, EN'de her isim ayrı düşünülür."
  ]
}'''

ORNEK_TEST_JSON = '''{
  "konu_id": 1,
  "konu_baslik": "Çoğul -s Testi",
  "test_sorulari": [
    {
      "soru": "Aşağıdakilerden hangisi doğrudur?",
      "secenekler": [
        "A) Writer write book",
        "B) Writers write books",
        "C) Writers writes books",
        "D) Writer writes book"
      ],
      "cevap": "B",
      "aciklama": "Çoğul özne ile fiil yalın, nesne de çoğul."
    }
  ]
}'''

ORNEK_ALISTIRMA_JSON = '''{
  "konu_id": 1,
  "konu_baslik": "Çoğul -s Karışık Alıştırmalar",
  "alistirmalar": [
    {
      "tip": "kelime_siralama",
      "soru": "Kelimeleri sırala: 'Doktorlar hasta tedavi eder'",
      "kelimeler": ["patients", "treat", "doctors"],
      "dogru_siralama": ["doctors", "treat", "patients"]
    },
    {
      "tip": "serbest_ceviri",
      "yon": "tr_to_en",
      "soru": "Çocuklar parkta oyun oynar.",
      "dogru_cevap": "Children play games in the park.",
      "alternatifler": ["Kids play games in the park", "Children play in the park"]
    }
  ]
}'''

YZ_PROMPT_TEMPLATE = '''Sen bir İngilizce öğretim materyali hazırlayıcısısın. Sana vereceğim YouTube
video linkindeki dersi izleyip, AŞAĞIDAKİ JSON ŞEMASINA TAM UYGUN bir ders 
dosyası üreteceksin.

VİDEO: [VIDEO_LINK]
KONU_ID: [KONU_ID]
EKSTRA NOT: [Video özelinde isteğin varsa yaz, yoksa "yok"]

KURALLAR:
1. Çıktı SADECE geçerli JSON olmalı. Açıklama, ```json bloğu YOK.
2. Tüm açıklamalar Türkçe, İngilizce örnekler doğal ve hocaya sadık.
3. Hoca videoda hangi yöntemi kullanıyorsa "alistirmalar" kısmı ona göre
   doldurulmalı. Hocanın imzası: önce TR mantığında diz, sonra EN kelimeleri 
   yerleştir. Bu yöntem için "kelime_siralama" tipi kullan.
4. "kelime_listesi": videoda geçen TÜM önemli kelimeleri ekle, her birine
   tur (isim/fiil/sıfat/zarf), örnek_en ve örnek_tr ver.
5. "alistirmalar": en az 6, ideal 8-12 alıştırma. Tipler:
   - "kelime_siralama": kelimeler dizisi ve dogru_siralama listesi
   - "bosluk_doldurma": cumle (___ ile boşluk), bosluklar [{ipucu, dogru_cevap, alternatifler}]
   - "serbest_ceviri": yon (tr_to_en/en_to_tr), soru, dogru_cevap, alternatifler
   - "coktan_secmeli": secenekler [A) ..., B) ...], cevap (sadece harf)
6. "ornek_cumleler": hocanın videoda verdiği örnekleri birebir koy, vurgu ekle.
7. "hatirlatmalar": kuralın 3-5 maddelik özeti.
8. Eğer videoda bir blok için içerik yoksa boş dizi [] bırak, uydurma. Ama
   "ders_icerik" ve "kelime_listesi" her zaman dolu olmalı.
9. "seviye": "başlangıç" / "orta" / "ileri".
10. "video_suresi": "MM:SS" formatında.
11. Çeviri alıştırmalarında "alternatifler" alanına makul varyasyonları ekle
    (artikelli/artikelsiz, eş anlamlı, vs.) — uygulama fuzzy match yapıyor ama
    alternatifler garantili kabul.

ŞEMA ÖRNEĞİ (bunu format referansı olarak kullan):
[Buraya yukarıdaki ORNEK_DERS_JSON'u yapıştır]

Şimdi bu şemaya uygun JSON üret.'''

# ============================================================
# SIDEBAR NAVİGASYON
# ============================================================
st.sidebar.title("🇬🇧 English Journey")
pages = {
    "🏠 Ana Sayfa": "ana",
    "📖 Dersler": "dersler",
    "🎯 Testler": "testler",
    "🧩 Alıştırmalar": "alistirmalar",
    "📊 İlerleme": "ilerleme",
    "⚙️ Ayarlar": "ayarlar"
}
selected = st.sidebar.radio("Sayfa:", list(pages.keys()))

# Ana sayfa
if selected == "🏠 Ana Sayfa":
    st.markdown("<h1 class='main-header'>🇬🇧 English Journey'e Hoş Geldin!</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Toplam Ders", len(tum_dersleri_yukle()))
    with col2:
        st.metric("🎯 Toplam Test", len(tum_testleri_yukle()))
    with col3:
        st.metric("🧩 Alıştırma Seti", len(tum_alistirmalari_yukle()))
    with col4:
        st.metric("⭐ Başarı Puanı", st.session_state.ilerleme.get('basari_puani', 0))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **📖 İnteraktif Dersler**
        - YouTube video eşliğinde
        - Gramer + kelime + alıştırma birlikte
        - TR ↔ EN mantık farklarına vurgu
        """)
        st.info("""
        **🎯 Testler**
        - Kavram pekiştirme
        - Çoktan seçmeli
        - Anında geri bildirim
        """)
    with col2:
        st.info("""
        **🧩 4 Tip Alıştırma**
        - 🔤 Kelime sıralama (hocanın yöntemi!)
        - 🔲 Boşluk doldurma
        - ✍️ Serbest çeviri (fuzzy match)
        - ☑️ Çoktan seçmeli
        """)
        st.info("""
        **📊 İlerleme + Yedek**
        - Başarı oranı takibi
        - Seviye sistemi
        - ZIP yedekleme/geri yükleme
        """)
    
    st.markdown("---")
    st.subheader("⚡ Hızlı Başlangıç")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📖 Derslere Git", use_container_width=True, type="primary"):
            st.session_state.current_page = "📖 Dersler"
            st.rerun()
    with col2:
        if st.button("🎯 Test Çöz", use_container_width=True):
            st.session_state.current_page = "🎯 Testler"
            st.rerun()
    with col3:
        if st.button("🧩 Alıştırma", use_container_width=True):
            st.session_state.current_page = "🧩 Alıştırmalar"
            st.rerun()
    with col4:
        if st.button("📊 İlerleme", use_container_width=True):
            st.session_state.current_page = "📊 İlerleme"
            st.rerun()
    
    st.markdown("---")
    
    # Özet
    tam = len(st.session_state.ilerleme['tamamlanan_dersler'])
    test = len(st.session_state.ilerleme['cozulen_testler'])
    alis = len(st.session_state.ilerleme['cozulen_alistirmalar'])
    
    if tam == 0 and test == 0 and alis == 0:
        st.info("🎯 **Henüz başlamadın!** İlk dersi açarak başla, JSON dosyalarını Ayarlar'dan yüklemeyi unutma.")
    else:
        st.success(f"""
        **✨ Süper gidiyorsun!**
        - ✅ {tam} ders tamamlandı
        - ✅ {test} test çözüldü
        - ✅ {alis} alıştırma seti tamamlandı
        """)
    
    st.info("💡 **İpucu:** Gemini'ye video linkini verip Ayarlar > Metin ile Yükleme'deki **YZ Prompt Template**'i kullan, JSON otomatik üretilir.")

elif selected == "📖 Dersler":
    dersler_sayfasi()
elif selected == "🎯 Testler":
    testler_sayfasi()
elif selected == "🧩 Alıştırmalar":
    alistirmalar_sayfasi()
elif selected == "📊 İlerleme":
    ilerleme_sayfasi()
elif selected == "⚙️ Ayarlar":
    ayarlar_sayfasi()

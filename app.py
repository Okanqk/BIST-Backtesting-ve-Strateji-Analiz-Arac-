import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# yfinance import kontrolü
try:
    import yfinance as yf
except ImportError:
    st.error("❌ yfinance kütüphanesi yüklü değil. Lütfen 'pip install yfinance' komutunu çalıştırın.")
    st.stop()

# Sayfa konfigürasyonu
st.set_page_config(page_title="BIST Backtesting Aracı", layout="wide", initial_sidebar_state="expanded")

# Başlık ve açıklama
st.title("📈 BIST Hisse Senedi Backtesting ve Strateji Analiz Aracı")
st.markdown("**BIST 100** hisseleri için teknik analiz ve strateji backtesting uygulaması")

# BIST 100 hisse listesi (popüler hisseler)
BIST_STOCKS = [
    "THYAO.IS", "SAHOL.IS", "EREGL.IS", "KCHOL.IS", "TUPRS.IS",
    "SISE.IS", "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS",
    "ASELS.IS", "BIMAS.IS", "TCELL.IS", "PETKM.IS", "KOZAL.IS",
    "SASA.IS", "PGSUS.IS", "AEFES.IS", "ARCLK.IS", "ENKAI.IS"
]

# Teknik indikatör hesaplama fonksiyonları
def calculate_ma(data, period):
    """Hareketli Ortalama hesapla"""
    return data['Close'].rolling(window=period).mean()

def calculate_rsi(data, period=14):
    """RSI (Relative Strength Index) hesapla"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD hesapla"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std=2):
    """Bollinger Bands hesapla"""
    ma = data['Close'].rolling(window=period).mean()
    std_dev = data['Close'].rolling(window=period).std()
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    return upper_band, ma, lower_band

# Strateji fonksiyonları
def ma_crossover_strategy(data, short_period, long_period):
    """MA Crossover stratejisi"""
    data['MA_Short'] = calculate_ma(data, short_period)
    data['MA_Long'] = calculate_ma(data, long_period)
    
    # Al/Sat sinyalleri
    data['Signal'] = 0
    data['Signal'][short_period:] = np.where(
        data['MA_Short'][short_period:] > data['MA_Long'][short_period:], 1, 0
    )
    data['Position'] = data['Signal'].diff()
    
    return data

def rsi_strategy(data, rsi_period=14, oversold=30, overbought=70):
    """RSI temelli strateji"""
    data['RSI'] = calculate_rsi(data, rsi_period)
    
    # Al/Sat sinyalleri
    data['Signal'] = 0
    data['Signal'] = np.where(data['RSI'] < oversold, 1, 0)  # Al sinyali
    data['Signal'] = np.where(data['RSI'] > overbought, -1, data['Signal'])  # Sat sinyali
    data['Position'] = data['Signal'].diff()
    
    return data

# Performans metrikleri hesaplama
def calculate_performance_metrics(data, initial_capital=100000):
    """Performans metriklerini hesapla"""
    # İşlem noktalarını bul
    buy_signals = data[data['Position'] == 1].index
    sell_signals = data[data['Position'] == -1].index
    
    # Portfolio değerini hesapla
    portfolio_value = initial_capital
    position = 0
    shares = 0
    portfolio_values = []
    
    for date in data.index:
        if date in buy_signals and position == 0:
            shares = portfolio_value / data.loc[date, 'Close']
            position = 1
        elif date in sell_signals and position == 1:
            portfolio_value = shares * data.loc[date, 'Close']
            shares = 0
            position = 0
        
        if position == 1:
            portfolio_values.append(shares * data.loc[date, 'Close'])
        else:
            portfolio_values.append(portfolio_value)
    
    data['Portfolio_Value'] = portfolio_values
    
    # Metrikler
    total_return = ((portfolio_values[-1] - initial_capital) / initial_capital) * 100
    
    # Sharpe Ratio
    returns = data['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win Rate
    trades = len(buy_signals)
    if trades > 0:
        profitable_trades = sum(1 for i in range(min(len(buy_signals), len(sell_signals)))
                               if data.loc[sell_signals[i], 'Close'] > data.loc[buy_signals[i], 'Close'])
        win_rate = (profitable_trades / trades) * 100 if trades > 0 else 0
    else:
        win_rate = 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades
    }

# Sidebar - Kullanıcı Girişleri
st.sidebar.header("⚙️ Ayarlar")

# Hisse seçimi
selected_stock = st.sidebar.selectbox(
    "Hisse Senedi Seçin:",
    BIST_STOCKS,
    index=0
)

# Tarih aralığı
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Başlangıç Tarihi",
        value=datetime.now() - timedelta(days=365*2)
    )
with col2:
    end_date = st.date_input(
        "Bitiş Tarihi",
        value=datetime.now()
    )

# Strateji seçimi
st.sidebar.subheader("📊 Strateji Seçimi")
strategy = st.sidebar.radio(
    "Strateji:",
    ["MA Crossover", "RSI Stratejisi"]
)

# Strateji parametreleri
if strategy == "MA Crossover":
    st.sidebar.subheader("MA Crossover Parametreleri")
    short_ma = st.sidebar.slider("Kısa MA Periyodu", 5, 50, 20)
    long_ma = st.sidebar.slider("Uzun MA Periyodu", 20, 200, 50)
else:
    st.sidebar.subheader("RSI Stratejisi Parametreleri")
    rsi_period = st.sidebar.slider("RSI Periyodu", 5, 30, 14)
    oversold = st.sidebar.slider("Aşırı Satım Seviyesi", 20, 40, 30)
    overbought = st.sidebar.slider("Aşırı Alım Seviyesi", 60, 80, 70)

# Başlangıç sermayesi
initial_capital = st.sidebar.number_input(
    "Başlangıç Sermayesi (TL)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000
)

# Veri çekme butonu
if st.sidebar.button("🚀 Analizi Başlat", type="primary"):
    try:
        with st.spinner(f"{selected_stock} verisi çekiliyor..."):
            # Veri çekme
            data = yf.download(selected_stock, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error("❌ Veri çekilemedi. Lütfen farklı bir hisse veya tarih aralığı deneyin.")
            else:
                # Teknik indikatörleri hesapla
                data['MA_20'] = calculate_ma(data, 20)
                data['MA_50'] = calculate_ma(data, 50)
                data['MA_200'] = calculate_ma(data, 200)
                data['RSI'] = calculate_rsi(data, 14)
                data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)
                data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)
                
                # Seçilen stratejiyi uygula
                if strategy == "MA Crossover":
                    data = ma_crossover_strategy(data, short_ma, long_ma)
                    strategy_name = f"MA Crossover ({short_ma}/{long_ma})"
                else:
                    data = rsi_strategy(data, rsi_period, oversold, overbought)
                    strategy_name = f"RSI Stratejisi (RSI: {rsi_period}, OS: {oversold}, OB: {overbought})"
                
                # Performans metriklerini hesapla
                metrics = calculate_performance_metrics(data, initial_capital)
                
                # Sonuçları göster
                st.success(f"✅ {selected_stock} analizi tamamlandı!")
                
                # Performans Metrikleri
                st.header("📊 Performans Metrikleri")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Toplam Getiri", f"%{metrics['total_return']:.2f}")
                with col2:
                    st.metric("Sharpe Oranı", f"{metrics['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Maks. Düşüş", f"%{metrics['max_drawdown']:.2f}")
                with col4:
                    st.metric("Kazanma Oranı", f"%{metrics['win_rate']:.2f}")
                with col5:
                    st.metric("İşlem Sayısı", f"{metrics['trades']}")
                
                # Grafikler
                st.header("📈 Grafikler ve Analiz")
                
                # Ana grafik (Fiyat + İndikatörler)
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        f'{selected_stock} - Fiyat ve Hareketli Ortalamalar',
                        'RSI',
                        'MACD',
                        'Volume'
                    ),
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
                
                # Fiyat ve MA'lar
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Fiyat'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MA_200'], name='MA 200', line=dict(color='red', width=1)),
                    row=1, col=1
                )
                
                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Üst', 
                              line=dict(color='gray', width=1, dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Alt',
                              line=dict(color='gray', width=1, dash='dash'), fill='tonexty'),
                    row=1, col=1
                )
                
                # Al/Sat sinyalleri
                buy_signals = data[data['Position'] == 1]
                sell_signals = data[data['Position'] == -1]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index, y=buy_signals['Close'],
                        mode='markers', name='AL',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index, y=sell_signals['Close'],
                        mode='markers', name='SAT',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ),
                    row=1, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='orange')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color='gray'),
                    row=3, col=1
                )
                
                # Volume
                colors = ['red' if data['Close'][i] < data['Open'][i] else 'green' 
                         for i in range(len(data))]
                fig.add_trace(
                    go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
                    row=4, col=1
                )
                
                # Layout
                fig.update_layout(
                    height=1200,
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(title_text="Tarih", row=4, col=1)
                fig.update_yaxes(title_text="Fiyat (TL)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                fig.update_yaxes(title_text="MACD", row=3, col=1)
                fig.update_yaxes(title_text="Volume", row=4, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Equity Curve
                st.header("💰 Portfolio Değer Eğrisi")
                fig_equity = go.Figure()
                fig_equity.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Portfolio_Value'],
                        name='Portfolio Değeri',
                        line=dict(color='green', width=2)
                    )
                )
                fig_equity.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Başlangıç Sermayesi"
                )
                fig_equity.update_layout(
                    title=f"Portfolio Değer Eğrisi - {strategy_name}",
                    xaxis_title="Tarih",
                    yaxis_title="Portfolio Değeri (TL)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # İşlem detayları
                st.header("📋 İşlem Detayları")
                trades_data = []
                buy_list = buy_signals.index.tolist()
                sell_list = sell_signals.index.tolist()
                
                for i in range(min(len(buy_list), len(sell_list))):
                    buy_price = data.loc[buy_list[i], 'Close']
                    sell_price = data.loc[sell_list[i], 'Close']
                    profit = ((sell_price - buy_price) / buy_price) * 100
                    
                    trades_data.append({
                        'Alış Tarihi': buy_list[i].strftime('%Y-%m-%d'),
                        'Alış Fiyatı': f"{buy_price:.2f} TL",
                        'Satış Tarihi': sell_list[i].strftime('%Y-%m-%d'),
                        'Satış Fiyatı': f"{sell_price:.2f} TL",
                        'Kar/Zarar': f"%{profit:.2f}"
                    })
                
                if trades_data:
                    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
                else:
                    st.info("Seçilen dönemde tamamlanmış işlem bulunmamaktadır.")
                
                # Özet bilgi
                st.header("ℹ️ Analiz Özeti")
                st.info(f"""
                **Hisse:** {selected_stock}  
                **Strateji:** {strategy_name}  
                **Analiz Dönemi:** {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}  
                **Veri Noktası Sayısı:** {len(data)}  
                **Başlangıç Sermayesi:** {initial_capital:,.0f} TL  
                **Kapanış Portfolio Değeri:** {data['Portfolio_Value'].iloc[-1]:,.2f} TL
                """)
                
    except Exception as e:
        st.error(f"❌ Bir hata oluştu: {str(e)}")
        st.info("Lütfen farklı bir hisse veya tarih aralığı deneyin.")

# Bilgilendirme
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Kullanım Kılavuzu
1. **Hisse Seçin:** BIST 100'den bir hisse seçin
2. **Tarih Aralığı:** Analiz yapmak istediğiniz dönemi seçin
3. **Strateji Seçin:** MA Crossover veya RSI stratejisi
4. **Parametreleri Ayarlayın:** Strateji parametrelerini optimize edin
5. **Analizi Başlatın:** Butona tıklayın ve sonuçları inceleyin

### ⚠️ Önemli Notlar
- Bu araç sadece eğitim amaçlıdır
- Geçmiş performans gelecek getiriyi garanti etmez
- Yatırım kararlarınızı profesyonel danışmanlık alarak verin
""")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Geliştirici Notu:** BIST verileri için .IS uzantısı kullanılmaktadır.")

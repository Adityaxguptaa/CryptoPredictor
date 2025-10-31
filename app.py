import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from xgboost import XGBClassifier
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(page_title="Bitcoin Sentiment & Trader Performance ML Analytics", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    fear_greed = pd.read_csv('attached_assets/fear_greed_index_1761747110179.csv')
    trader_data = pd.read_csv('attached_assets/historical_data_1761747110176.csv')
    
    fear_greed['date'] = pd.to_datetime(fear_greed['date'])
    trader_data['Timestamp IST'] = pd.to_datetime(trader_data['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trader_data['date'] = trader_data['Timestamp IST'].dt.date
    trader_data['date'] = pd.to_datetime(trader_data['date'])
    
    return fear_greed, trader_data

@st.cache_data
def preprocess_and_merge(fear_greed, trader_data):
    daily_trader_metrics = trader_data.groupby('date').agg({
        'Closed PnL': ['sum', 'mean', 'std', 'count'],
        'Size USD': ['sum', 'mean'],
        'Account': 'nunique',
        'Side': lambda x: (x == 'BUY').sum() / len(x) * 100
    }).reset_index()
    
    daily_trader_metrics.columns = ['date', 'total_pnl', 'avg_pnl', 'std_pnl', 'num_trades',
                                      'total_volume', 'avg_trade_size', 'unique_accounts', 'buy_percentage']
    
    merged_data = pd.merge(fear_greed, daily_trader_metrics, on='date', how='inner')
    merged_data['profitable_day'] = (merged_data['total_pnl'] > 0).astype(int)
    
    trader_data['profitable_trade'] = (trader_data['Closed PnL'] > 0).astype(int)
    trader_account_data = pd.merge(trader_data, fear_greed[['date', 'value', 'classification']], 
                                    on='date', how='left')
    
    return merged_data, daily_trader_metrics, trader_account_data

@st.cache_data
def engineer_features(merged_data):
    merged_data['sentiment_numeric'] = merged_data['value']
    merged_data['is_extreme_fear'] = (merged_data['classification'] == 'Extreme Fear').astype(int)
    merged_data['is_fear'] = (merged_data['classification'] == 'Fear').astype(int)
    merged_data['is_neutral'] = (merged_data['classification'] == 'Neutral').astype(int)
    merged_data['is_greed'] = (merged_data['classification'] == 'Greed').astype(int)
    merged_data['is_extreme_greed'] = (merged_data['classification'] == 'Extreme Greed').astype(int)
    
    merged_data['pnl_per_trade'] = merged_data['total_pnl'] / merged_data['num_trades']
    merged_data['volume_per_account'] = merged_data['total_volume'] / merged_data['unique_accounts']
    
    merged_data['sentiment_ma_7'] = merged_data['sentiment_numeric'].rolling(window=7, min_periods=1).mean()
    merged_data['sentiment_ma_30'] = merged_data['sentiment_numeric'].rolling(window=30, min_periods=1).mean()
    merged_data['pnl_ma_7'] = merged_data['total_pnl'].rolling(window=7, min_periods=1).mean()
    
    merged_data['sentiment_change'] = merged_data['sentiment_numeric'].diff()
    merged_data['pnl_volatility'] = merged_data['std_pnl'].fillna(0)
    
    return merged_data

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division='warn'),
            'recall': recall_score(y_test, y_pred, zero_division='warn'),
            'f1': f1_score(y_test, y_pred, zero_division='warn'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        trained_models[name] = model
    
    return results, trained_models

def prepare_sequences(data, feature_cols, target_col, sequence_length=10):
    """Prepare sequences for LSTM/GRU models"""
    X, y = [], []
    data_sorted = data.sort_values('date').reset_index(drop=True)
    features = data_sorted[feature_cols].fillna(0).values
    targets = data_sorted[target_col].values
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    
    return np.array(X), np.array(y)

@st.cache_resource
def train_deep_learning_models(X_train, X_test, y_train, y_test, input_shape):
    """Train LSTM and GRU models"""
    models = {}
    histories = {}
    results = {}
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lstm_model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                                   validation_split=0.2, callbacks=[early_stop], verbose=0)
    
    gru_model = Sequential([
        GRU(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                                validation_split=0.2, callbacks=[early_stop], verbose=0)
    
    for name, model, history in [('LSTM', lstm_model, lstm_history), ('GRU', gru_model, gru_history)]:
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division='warn'),
            'recall': recall_score(y_test, y_pred, zero_division='warn'),
            'f1': f1_score(y_test, y_pred, zero_division='warn'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        models[name] = model
        histories[name] = history
    
    return results, models, histories

def perform_trader_clustering(trader_account_data, merged_data, n_clusters=4):
    """Perform trader segmentation using clustering"""
    trader_features = trader_account_data.groupby('Account').agg({
        'Closed PnL': ['sum', 'mean', 'std', 'count'],
        'Size USD': ['sum', 'mean'],
        'value': 'mean',
        'Side': lambda x: (x == 'BUY').sum() / len(x) * 100
    }).reset_index()
    
    trader_features.columns = ['Account', 'total_pnl', 'avg_pnl', 'pnl_std', 'num_trades',
                                'total_volume', 'avg_trade_size', 'avg_sentiment', 'buy_percentage']
    
    trader_features = trader_features[trader_features['num_trades'] >= 10]
    
    feature_cols = ['total_pnl', 'avg_pnl', 'num_trades', 'total_volume', 
                    'avg_trade_size', 'buy_percentage']
    X_cluster = trader_features[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    trader_features['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        trader_features['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    except:
        trader_features['cluster_dbscan'] = 0
    
    silhouette_avg = silhouette_score(X_scaled, trader_features['cluster_kmeans'])
    
    return trader_features, kmeans, silhouette_avg, X_scaled

def run_backtest(merged_data, strategy='extreme_fear_buy'):
    """Run backtesting simulation"""
    df = merged_data.copy().sort_values('date').reset_index(drop=True)
    
    initial_capital = 100000
    capital = initial_capital
    position = 0
    entry_value = 0
    trades = []
    portfolio_value = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if strategy == 'extreme_fear_buy':
            if row['classification'] == 'Extreme Fear' and position == 0:
                entry_value = capital
                position = capital
                capital = 0
                trades.append({'date': row['date'], 'action': 'BUY', 'sentiment': row['value'], 
                              'value': entry_value, 'pnl': 0, 'profitable': None})
            elif row['classification'] in ['Greed', 'Extreme Greed'] and position > 0:
                pnl_pct = row['total_pnl'] / max(abs(row['total_volume']), 1) if row['total_volume'] != 0 else 0
                exit_value = position * (1 + pnl_pct)
                trade_pnl = exit_value - entry_value
                capital = exit_value
                position = 0
                trades.append({'date': row['date'], 'action': 'SELL', 'sentiment': row['value'], 
                              'value': exit_value, 'pnl': trade_pnl, 'profitable': trade_pnl > 0})
        
        elif strategy == 'sentiment_momentum':
            if row['sentiment_change'] > 10 and position == 0:
                entry_value = capital
                position = capital
                capital = 0
                trades.append({'date': row['date'], 'action': 'BUY', 'sentiment': row['value'], 
                              'value': entry_value, 'pnl': 0, 'profitable': None})
            elif row['sentiment_change'] < -10 and position > 0:
                pnl_pct = row['total_pnl'] / max(abs(row['total_volume']), 1) if row['total_volume'] != 0 else 0
                exit_value = position * (1 + pnl_pct)
                trade_pnl = exit_value - entry_value
                capital = exit_value
                position = 0
                trades.append({'date': row['date'], 'action': 'SELL', 'sentiment': row['value'], 
                              'value': exit_value, 'pnl': trade_pnl, 'profitable': trade_pnl > 0})
        
        current_value = capital + position
        portfolio_value.append(current_value)
    
    if position > 0:
        capital += position
    
    final_capital = capital
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    portfolio_values = pd.Series(portfolio_value)
    returns = portfolio_values.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = (portfolio_values / initial_capital - 1) * 100
    max_drawdown = (cumulative - cumulative.cummax()).min()
    
    win_trades = sum(1 for trade in trades if trade.get('profitable') == True)
    total_completed_trades = sum(1 for trade in trades if trade.get('profitable') is not None)
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'completed_trades': total_completed_trades,
        'win_trades': win_trades,
        'trades': trades,
        'portfolio_value': portfolio_value,
        'dates': df['date'].tolist()
    }

def generate_pdf_report(merged_data, model_results, insights):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                  fontSize=24, textColor=colors.HexColor('#1f77b4'), 
                                  spaceAfter=30, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                    fontSize=16, textColor=colors.HexColor('#2ca02c'), 
                                    spaceAfter=12)
    
    story.append(Paragraph("Bitcoin Sentiment & Trader Performance", title_style))
    story.append(Paragraph("ML Analytics Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(f"Analysis Period: {merged_data['date'].min().strftime('%Y-%m-%d')} to {merged_data['date'].max().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Paragraph(f"Total Trading Days: {len(merged_data):,}", styles['Normal']))
    story.append(Paragraph(f"Overall Win Rate: {(merged_data['profitable_day'].mean()*100):.1f}%", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Model Performance Summary", heading_style))
    perf_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    for model_name, results in model_results.items():
        perf_data.append([
            model_name,
            f"{results['accuracy']:.3f}",
            f"{results['precision']:.3f}",
            f"{results['recall']:.3f}",
            f"{results['f1']:.3f}",
            f"{results['roc_auc']:.3f}"
        ])
    
    table = Table(perf_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Key Insights", heading_style))
    for insight in insights:
        story.append(Paragraph(f"• {insight}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    story.append(Paragraph("Recommendations", heading_style))
    story.append(Paragraph("Based on the analysis, consider the following trading strategies:", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    best_sentiment = merged_data.groupby('classification')['total_pnl'].mean().idxmax()
    story.append(Paragraph(f"1. Focus trading during {best_sentiment} periods for optimal performance", styles['Normal']))
    story.append(Paragraph("2. Use ML models for daily profitability predictions", styles['Normal']))
    story.append(Paragraph("3. Monitor sentiment shifts for trading opportunities", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.title("🚀 Bitcoin Market Sentiment vs Trader Performance ML Analytics")
    st.markdown("### Exploring the relationship between Fear/Greed Index and Hyperliquid trading outcomes")
    
    with st.spinner("Loading datasets..."):
        fear_greed, trader_data = load_data()
        merged_data, daily_trader_metrics, trader_account_data = preprocess_and_merge(fear_greed, trader_data)
        merged_data = engineer_features(merged_data)
    
    tabs = st.tabs(["📊 Data Overview", "📈 Exploratory Analysis", "🤖 ML Model Training", 
                    "📉 Model Evaluation", "🔬 Statistical Insights", "🎯 Predictions",
                    "🧠 Deep Learning Models", "👥 Trader Segmentation", "💹 Backtesting", "📄 PDF Report"])
    
    with tabs[0]:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trading Days", f"{len(merged_data):,}")
        with col2:
            st.metric("Total Trades", f"{trader_data.shape[0]:,}")
        with col3:
            st.metric("Unique Traders", f"{trader_data['Account'].nunique():,}")
        with col4:
            st.metric("Date Range", f"{merged_data['date'].min().strftime('%Y-%m-%d')} to {merged_data['date'].max().strftime('%Y-%m-%d')}")
        
        st.subheader("Dataset Samples")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fear/Greed Index Data**")
            st.dataframe(fear_greed.head(10), use_container_width=True)
            
            sentiment_dist = fear_greed['classification'].value_counts()
            fig = px.pie(values=sentiment_dist.values, names=sentiment_dist.index,
                        title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Trader Data Sample**")
            st.dataframe(trader_data.head(10), use_container_width=True)
            
            side_dist = trader_data['Side'].value_counts()
            fig = px.pie(values=side_dist.values, names=side_dist.index,
                        title='Buy vs Sell Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Merged Dataset with Features")
        st.dataframe(merged_data.head(20), use_container_width=True)
        st.write(f"**Shape:** {merged_data.shape[0]} rows × {merged_data.shape[1]} columns")
    
    with tabs[1]:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Time Series: Sentiment vs Trading Performance")
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Fear/Greed Index Over Time', 'Daily Total PnL Over Time'),
                           vertical_spacing=0.12)
        
        fig.add_trace(go.Scatter(x=merged_data['date'], y=merged_data['value'],
                                mode='lines', name='Sentiment Value',
                                line=dict(color='blue', width=2)), row=1, col=1)
        
        colors = merged_data['total_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
        fig.add_trace(go.Bar(x=merged_data['date'], y=merged_data['total_pnl'],
                            name='Daily PnL', marker_color=colors), row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Value", row=1, col=1)
        fig.update_yaxes(title_text="Total PnL", row=2, col=1)
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("PnL Distribution by Sentiment Classification")
        
        pnl_by_sentiment = merged_data.groupby('classification').agg({
            'total_pnl': ['mean', 'median', 'sum', 'count'],
            'profitable_day': 'mean'
        }).round(2)
        pnl_by_sentiment.columns = ['Avg PnL', 'Median PnL', 'Total PnL', 'Days', 'Profitable Day %']
        pnl_by_sentiment['Profitable Day %'] = (pnl_by_sentiment['Profitable Day %'] * 100).round(1)
        
        st.dataframe(pnl_by_sentiment, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(merged_data, x='classification', y='total_pnl',
                        title='PnL Distribution by Sentiment',
                        color='classification')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            win_rate_by_sentiment = merged_data.groupby('classification')['profitable_day'].mean() * 100
            fig = px.bar(x=win_rate_by_sentiment.index, y=win_rate_by_sentiment.values,
                        title='Win Rate (%) by Sentiment',
                        labels={'x': 'Sentiment', 'y': 'Win Rate (%)'},
                        color=win_rate_by_sentiment.values,
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Trading Metrics by Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_trades = merged_data.groupby('classification')['num_trades'].mean()
            fig = px.bar(x=avg_trades.index, y=avg_trades.values,
                        title='Average Number of Trades by Sentiment',
                        labels={'x': 'Sentiment', 'y': 'Avg Trades'},
                        color=avg_trades.values)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_volume = merged_data.groupby('classification')['total_volume'].mean()
            fig = px.bar(x=avg_volume.index, y=avg_volume.values,
                        title='Average Trading Volume by Sentiment',
                        labels={'x': 'Sentiment', 'y': 'Avg Volume ($)'},
                        color=avg_volume.values)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        numeric_cols = ['value', 'total_pnl', 'avg_pnl', 'num_trades', 'total_volume', 
                       'avg_trade_size', 'buy_percentage', 'sentiment_ma_7', 'pnl_ma_7']
        corr_matrix = merged_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       aspect='auto',
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Machine Learning Model Training")
        st.markdown("**Objective:** Predict whether a trading day will be profitable based on market sentiment and trading patterns")
        
        feature_cols = ['sentiment_numeric', 'is_extreme_fear', 'is_fear', 'is_greed', 'is_extreme_greed',
                       'num_trades', 'total_volume', 'avg_trade_size', 'buy_percentage',
                       'sentiment_ma_7', 'sentiment_ma_30', 'sentiment_change', 'pnl_volatility']
        
        X = merged_data[feature_cols].fillna(0)
        y = merged_data['profitable_day']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.subheader("Dataset Split")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Set", f"{len(X_train)} samples")
        with col2:
            st.metric("Test Set", f"{len(X_test)} samples")
        with col3:
            st.metric("Features", f"{len(feature_cols)}")
        
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            train_dist = pd.Series(y_train).value_counts()
            fig = px.pie(values=train_dist.values, names=['Unprofitable', 'Profitable'],
                        title='Training Set Class Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            test_dist = pd.Series(y_test).value_counts()
            fig = px.pie(values=test_dist.values, names=['Unprofitable', 'Profitable'],
                        title='Test Set Class Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with st.spinner("Training models..."):
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.success("✅ All models trained successfully!")
        
        st.subheader("Model Performance Comparison")
        
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'Precision': [results[m]['precision'] for m in results],
            'Recall': [results[m]['recall'] for m in results],
            'F1-Score': [results[m]['f1'] for m in results],
            'ROC-AUC': [results[m]['roc_auc'] for m in results]
        }).round(4)
        
        st.dataframe(performance_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='lightgreen'), use_container_width=True)
        
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=performance_df['Model'], y=performance_df[metric]))
        
        fig.update_layout(barmode='group', title='Model Performance Metrics Comparison',
                         yaxis_title='Score', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state['models'] = trained_models
        st.session_state['results'] = results
        st.session_state['scaler'] = scaler
        st.session_state['feature_cols'] = feature_cols
        st.session_state['X_test'] = X_test_scaled
        st.session_state['y_test'] = y_test
    
    with tabs[3]:
        st.header("Model Evaluation & Diagnostics")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'ML Model Training' tab")
        else:
            results = st.session_state['results']
            trained_models = st.session_state['models']
            
            selected_model = st.selectbox("Select Model for Detailed Evaluation", list(results.keys()))
            
            model_results = results[selected_model]
            
            st.subheader(f"Confusion Matrix - {selected_model}")
            
            cm = model_results['confusion_matrix']
            
            fig = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Unprofitable', 'Profitable'],
                           y=['Unprofitable', 'Profitable'],
                           text_auto=True,
                           color_continuous_scale='Blues')
            fig.update_layout(title=f'Confusion Matrix - {selected_model}')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Negatives", int(cm[0, 0]))
            with col2:
                st.metric("False Positives", int(cm[0, 1]))
            with col3:
                st.metric("False Negatives", int(cm[1, 0]))
            with col4:
                st.metric("True Positives", int(cm[1, 1]))
            
            st.subheader("ROC Curve")
            
            fig = go.Figure()
            
            for model_name in results:
                fpr, tpr, _ = roc_curve(st.session_state['y_test'], results[model_name]['y_pred_proba'])
                auc_score = results[model_name]['roc_auc']
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                        name=f'{model_name} (AUC = {auc_score:.3f})'))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    name='Random Classifier', line=dict(dash='dash', color='gray')))
            
            fig.update_layout(title='ROC Curves - All Models',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Importance")
            
            if selected_model in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
                model = trained_models[selected_model]
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state['feature_cols'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature',
                            orientation='h',
                            title=f'Feature Importance - {selected_model}',
                            color='Importance',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(feature_importance, use_container_width=True)
            else:
                st.info("Feature importance is only available for tree-based models (Random Forest, Gradient Boosting, XGBoost)")
            
            st.subheader("Classification Report")
            y_pred = model_results['y_pred']
            report = classification_report(st.session_state['y_test'], y_pred, 
                                          target_names=['Unprofitable', 'Profitable'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    with tabs[4]:
        st.header("Statistical Insights & Pattern Detection")
        
        st.subheader("Correlation Analysis: Sentiment vs Trading Outcomes")
        
        sentiment_pnl_corr = merged_data['value'].corr(merged_data['total_pnl'])
        sentiment_trades_corr = merged_data['value'].corr(merged_data['num_trades'])
        sentiment_volume_corr = merged_data['value'].corr(merged_data['total_volume'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment ↔ PnL Correlation", f"{sentiment_pnl_corr:.4f}")
        with col2:
            st.metric("Sentiment ↔ Trade Count Correlation", f"{sentiment_trades_corr:.4f}")
        with col3:
            st.metric("Sentiment ↔ Volume Correlation", f"{sentiment_volume_corr:.4f}")
        
        fig = px.scatter(merged_data, x='value', y='total_pnl',
                        trendline='ols',
                        labels={'value': 'Sentiment Value', 'total_pnl': 'Total Daily PnL'},
                        title='Sentiment vs Daily PnL (with trend line)',
                        color='classification')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Hypothesis Testing: PnL Differences Between Sentiment Groups")
        
        extreme_fear_pnl = merged_data[merged_data['classification'] == 'Extreme Fear']['total_pnl']
        extreme_greed_pnl = merged_data[merged_data['classification'] == 'Extreme Greed']['total_pnl']
        fear_pnl = merged_data[merged_data['classification'] == 'Fear']['total_pnl']
        greed_pnl = merged_data[merged_data['classification'] == 'Greed']['total_pnl']
        
        if len(extreme_fear_pnl) > 0 and len(extreme_greed_pnl) > 0:
            t_stat, p_value = stats.ttest_ind(extreme_fear_pnl, extreme_greed_pnl)
            
            st.write("**T-Test: Extreme Fear vs Extreme Greed PnL**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("T-Statistic", f"{t_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.6f}")
            
            if p_value < 0.05:
                st.success(f"✅ **Statistically significant difference** (p < 0.05): Trading outcomes differ significantly between Extreme Fear and Extreme Greed periods")
            else:
                st.info(f"ℹ️ No statistically significant difference (p >= 0.05)")
        
        st.subheader("Key Insights & Patterns")
        
        insights = []
        
        best_sentiment = pnl_by_sentiment['Avg PnL'].idxmax()
        worst_sentiment = pnl_by_sentiment['Avg PnL'].idxmin()
        insights.append(f"📊 **Best performing sentiment:** {best_sentiment} (Avg PnL: ${pnl_by_sentiment.loc[best_sentiment, 'Avg PnL']:,.2f})")
        insights.append(f"📉 **Worst performing sentiment:** {worst_sentiment} (Avg PnL: ${pnl_by_sentiment.loc[worst_sentiment, 'Avg PnL']:,.2f})")
        
        highest_win_rate_sentiment = win_rate_by_sentiment.idxmax()
        insights.append(f"🎯 **Highest win rate:** {highest_win_rate_sentiment} ({win_rate_by_sentiment.max():.1f}% of days profitable)")
        
        total_profitable_days = merged_data['profitable_day'].sum()
        total_days = len(merged_data)
        overall_win_rate = (total_profitable_days / total_days) * 100
        insights.append(f"📈 **Overall win rate:** {overall_win_rate:.1f}% ({total_profitable_days}/{total_days} days)")
        
        avg_sentiment = merged_data['value'].mean()
        if avg_sentiment < 25:
            market_mood = "Extreme Fear"
        elif avg_sentiment < 45:
            market_mood = "Fear"
        elif avg_sentiment < 55:
            market_mood = "Neutral"
        elif avg_sentiment < 75:
            market_mood = "Greed"
        else:
            market_mood = "Extreme Greed"
        insights.append(f"😊 **Average market sentiment:** {avg_sentiment:.1f} ({market_mood})")
        
        for insight in insights:
            st.markdown(insight)
        
        st.subheader("Trading Volume Analysis by Sentiment")
        
        fig = px.violin(merged_data, x='classification', y='total_volume',
                       box=True, points='all',
                       title='Trading Volume Distribution by Sentiment',
                       color='classification')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Buy Pressure Analysis")
        
        buy_pressure = merged_data.groupby('classification')['buy_percentage'].mean().sort_values(ascending=False)
        fig = px.bar(x=buy_pressure.index, y=buy_pressure.values,
                    title='Average Buy Percentage by Sentiment',
                    labels={'x': 'Sentiment', 'y': 'Buy Percentage (%)'},
                    color=buy_pressure.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        st.header("Interactive Prediction Interface")
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'ML Model Training' tab")
        else:
            st.subheader("Predict Trading Day Profitability")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_value = st.slider("Sentiment Value (0-100)", 0, 100, 50)
                num_trades = st.number_input("Number of Trades", min_value=1, max_value=10000, value=100)
                total_volume = st.number_input("Total Volume ($)", min_value=0, max_value=10000000, value=100000)
                avg_trade_size = st.number_input("Average Trade Size ($)", min_value=0, max_value=1000000, value=1000)
                buy_percentage = st.slider("Buy Percentage (%)", 0, 100, 50)
            
            with col2:
                sentiment_ma_7 = st.slider("7-Day Sentiment MA", 0, 100, sentiment_value)
                sentiment_ma_30 = st.slider("30-Day Sentiment MA", 0, 100, sentiment_value)
                sentiment_change = st.slider("Sentiment Change", -50, 50, 0)
                pnl_volatility = st.number_input("PnL Volatility", min_value=0.0, max_value=100000.0, value=1000.0)
            
            is_extreme_fear = 1 if sentiment_value < 25 else 0
            is_fear = 1 if 25 <= sentiment_value < 45 else 0
            is_greed = 1 if 55 <= sentiment_value < 75 else 0
            is_extreme_greed = 1 if sentiment_value >= 75 else 0
            
            input_features = np.array([[
                sentiment_value, is_extreme_fear, is_fear, is_greed, is_extreme_greed,
                num_trades, total_volume, avg_trade_size, buy_percentage,
                sentiment_ma_7, sentiment_ma_30, sentiment_change, pnl_volatility
            ]])
            
            input_scaled = st.session_state['scaler'].transform(input_features)
            
            selected_pred_model = st.selectbox("Select Model for Prediction", list(st.session_state['models'].keys()))
            
            if st.button("🔮 Predict", type="primary"):
                model = st.session_state['models'][selected_pred_model]
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("✅ **Predicted Outcome: PROFITABLE DAY**")
                    else:
                        st.error("❌ **Predicted Outcome: UNPROFITABLE DAY**")
                
                with col2:
                    st.metric("Confidence (Profitable)", f"{prediction_proba[1]*100:.1f}%")
                    st.metric("Confidence (Unprofitable)", f"{prediction_proba[0]*100:.1f}%")
                
                fig = go.Figure(go.Bar(
                    x=['Unprofitable', 'Profitable'],
                    y=[prediction_proba[0]*100, prediction_proba[1]*100],
                    marker_color=['red', 'green'],
                    text=[f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"],
                    textposition='auto'
                ))
                fig.update_layout(title='Prediction Probability Distribution',
                                yaxis_title='Probability (%)',
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Model Used:** {selected_pred_model}")
    
    with tabs[6]:
        st.header("Deep Learning Sequential Models (LSTM & GRU)")
        st.markdown("**Advanced time-series prediction using recurrent neural networks**")
        
        sequence_length = st.slider("Sequence Length (days)", 5, 30, 10)
        
        feature_cols_dl = ['sentiment_numeric', 'num_trades', 'total_volume', 'avg_trade_size', 
                           'buy_percentage', 'sentiment_ma_7']
        
        with st.spinner("Preparing sequences for deep learning..."):
            X_seq, y_seq = prepare_sequences(merged_data, feature_cols_dl, 'profitable_day', sequence_length)
            
            if len(X_seq) < 50:
                st.warning("⚠️ Not enough data for deep learning models. Need at least 50 sequences.")
            else:
                train_size = int(0.8 * len(X_seq))
                X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
                y_train_seq, y_test_seq = y_seq[:train_size], y_seq[train_size:]
                
                st.subheader("Sequence Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Sequences", len(X_train_seq))
                with col2:
                    st.metric("Test Sequences", len(X_test_seq))
                with col3:
                    st.metric("Sequence Shape", f"{X_train_seq.shape[1]} × {X_train_seq.shape[2]}")
                
                if st.button("🧠 Train Deep Learning Models", type="primary"):
                    with st.spinner("Training LSTM and GRU models... This may take a few minutes."):
                        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
                        dl_results, dl_models, dl_histories = train_deep_learning_models(
                            X_train_seq, X_test_seq, y_train_seq, y_test_seq, input_shape
                        )
                        
                        st.session_state['dl_results'] = dl_results
                        st.session_state['dl_models'] = dl_models
                        st.session_state['dl_histories'] = dl_histories
                    
                    st.success("✅ Deep learning models trained successfully!")
                
                if 'dl_results' in st.session_state:
                    st.subheader("Deep Learning Model Performance")
                    
                    dl_perf_df = pd.DataFrame({
                        'Model': list(st.session_state['dl_results'].keys()),
                        'Accuracy': [st.session_state['dl_results'][m]['accuracy'] for m in st.session_state['dl_results']],
                        'Precision': [st.session_state['dl_results'][m]['precision'] for m in st.session_state['dl_results']],
                        'Recall': [st.session_state['dl_results'][m]['recall'] for m in st.session_state['dl_results']],
                        'F1-Score': [st.session_state['dl_results'][m]['f1'] for m in st.session_state['dl_results']],
                        'ROC-AUC': [st.session_state['dl_results'][m]['roc_auc'] for m in st.session_state['dl_results']]
                    }).round(4)
                    
                    st.dataframe(dl_perf_df, use_container_width=True)
                    
                    if 'results' in st.session_state:
                        st.subheader("Model Comparison: Traditional ML vs Deep Learning")
                        
                        all_models_perf = []
                        for model_name, results in st.session_state['results'].items():
                            all_models_perf.append({
                                'Model': model_name,
                                'Type': 'Traditional ML',
                                'Accuracy': results['accuracy'],
                                'ROC-AUC': results['roc_auc']
                            })
                        
                        for model_name, results in st.session_state['dl_results'].items():
                            all_models_perf.append({
                                'Model': model_name,
                                'Type': 'Deep Learning',
                                'Accuracy': results['accuracy'],
                                'ROC-AUC': results['roc_auc']
                            })
                        
                        comparison_df = pd.DataFrame(all_models_perf)
                        
                        fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'ROC-AUC'],
                                    barmode='group', color_discrete_sequence=['#636EFA', '#EF553B'],
                                    title='Model Performance Comparison')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Training History")
                    selected_dl_model = st.selectbox("Select Model", list(st.session_state['dl_histories'].keys()))
                    
                    history = st.session_state['dl_histories'][selected_dl_model]
                    
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
                    
                    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'), row=1, col=1)
                    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'), row=1, col=1)
                    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'), row=1, col=2)
                    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'), row=1, col=2)
                    
                    fig.update_xaxes(title_text="Epoch", row=1, col=1)
                    fig.update_xaxes(title_text="Epoch", row=1, col=2)
                    fig.update_yaxes(title_text="Loss", row=1, col=1)
                    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[7]:
        st.header("Advanced Trader Segmentation")
        st.markdown("**Identify distinct trader personas using clustering algorithms**")
        
        n_clusters = st.slider("Number of Clusters (K-Means)", 2, 8, 4)
        
        if st.button("🔍 Perform Clustering Analysis", type="primary"):
            with st.spinner("Clustering traders..."):
                trader_features, kmeans_model, silhouette, X_scaled = perform_trader_clustering(
                    trader_account_data, merged_data, n_clusters
                )
                
                st.session_state['trader_features'] = trader_features
                st.session_state['kmeans_model'] = kmeans_model
                st.session_state['silhouette'] = silhouette
            
            st.success(f"✅ Identified {n_clusters} trader segments!")
        
        if 'trader_features' in st.session_state:
            trader_features = st.session_state['trader_features']
            
            st.subheader("Clustering Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{st.session_state['silhouette']:.3f}")
            with col2:
                st.metric("Number of Traders Analyzed", len(trader_features))
            
            st.subheader("Cluster Characteristics")
            
            cluster_summary = trader_features.groupby('cluster_kmeans').agg({
                'total_pnl': ['mean', 'sum'],
                'avg_pnl': 'mean',
                'num_trades': ['mean', 'sum'],
                'total_volume': 'mean',
                'buy_percentage': 'mean',
                'Account': 'count'
            }).round(2)
            
            cluster_summary.columns = ['Avg Total PnL', 'Sum PnL', 'Avg PnL/Trade', 
                                       'Avg Trades', 'Total Trades', 'Avg Volume', 
                                       'Buy %', 'Count']
            st.dataframe(cluster_summary, use_container_width=True)
            
            st.subheader("Cluster Visualization")
            
            fig = px.scatter(trader_features, x='total_pnl', y='num_trades',
                            color='cluster_kmeans', size='total_volume',
                            hover_data=['Account', 'avg_pnl', 'buy_percentage'],
                            title='Trader Segments (PnL vs Trade Count)',
                            labels={'cluster_kmeans': 'Cluster'})
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.box(trader_features, x='cluster_kmeans', y='avg_pnl',
                         color='cluster_kmeans',
                         title='PnL Distribution by Cluster',
                         labels={'cluster_kmeans': 'Cluster', 'avg_pnl': 'Average PnL'})
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Cluster Insights")
            for cluster_id in sorted(trader_features['cluster_kmeans'].unique()):
                cluster_data = trader_features[trader_features['cluster_kmeans'] == cluster_id]
                avg_pnl = cluster_data['total_pnl'].mean()
                trader_count = len(cluster_data)
                
                persona = "High-Volume Traders" if cluster_data['total_volume'].mean() > trader_features['total_volume'].median() else "Conservative Traders"
                if avg_pnl > 0:
                    performance = "Profitable"
                elif avg_pnl < 0:
                    performance = "Unprofitable"
                else:
                    performance = "Break-even"
                
                st.write(f"**Cluster {cluster_id}:** {trader_count} traders - {persona}, {performance} (Avg PnL: ${avg_pnl:,.2f})")
    
    with tabs[8]:
        st.header("Backtesting Framework")
        st.markdown("**Simulate trading strategies based on sentiment-driven predictions**")
        
        st.subheader("Strategy Selection")
        strategy_choice = st.selectbox("Select Trading Strategy", [
            "extreme_fear_buy",
            "sentiment_momentum"
        ])
        
        strategy_descriptions = {
            "extreme_fear_buy": "Buy during Extreme Fear periods, sell during Greed/Extreme Greed",
            "sentiment_momentum": "Buy when sentiment jumps +10 points, sell when it drops -10 points"
        }
        
        st.info(f"**Strategy:** {strategy_descriptions[strategy_choice]}")
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest simulation..."):
                backtest_results = run_backtest(merged_data, strategy=strategy_choice)
                st.session_state['backtest_results'] = backtest_results
            
            st.success("✅ Backtest completed!")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            st.subheader("Backtest Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Capital", f"${results['initial_capital']:,.0f}")
            with col2:
                st.metric("Final Capital", f"${results['final_capital']:,.0f}")
            with col3:
                return_color = "normal" if results['total_return'] >= 0 else "inverse"
                st.metric("Total Return", f"{results['total_return']:.2f}%", delta=f"{results['total_return']:.2f}%")
            with col4:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
            with col2:
                st.metric("Total Trades", results['num_trades'])
            with col3:
                st.metric("Completed Trades", results['completed_trades'])
            with col4:
                win_rate = (results['win_trades'] / max(results['completed_trades'], 1)) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                         delta=f"{results['win_trades']}/{results['completed_trades']} wins")
            
            st.subheader("Portfolio Value Over Time")
            
            portfolio_df = pd.DataFrame({
                'Date': results['dates'],
                'Portfolio Value': results['portfolio_value']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_df['Date'], y=portfolio_df['Portfolio Value'],
                                    mode='lines', name='Portfolio Value',
                                    line=dict(color='green' if results['total_return'] > 0 else 'red', width=2)))
            fig.add_hline(y=results['initial_capital'], line_dash="dash", 
                         annotation_text="Initial Capital", line_color="blue")
            fig.update_layout(title='Portfolio Value Over Time',
                            xaxis_title='Date', yaxis_title='Portfolio Value ($)',
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Trade History")
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed with this strategy")
    
    with tabs[9]:
        st.header("PDF Report Generation")
        st.markdown("**Export comprehensive analysis report with insights and recommendations**")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please train models first in the 'ML Model Training' tab")
        else:
            st.subheader("Report Configuration")
            
            st.write("The report will include:")
            st.write("- Executive summary with key metrics")
            st.write("- Model performance comparison table")
            st.write("- Statistical insights and patterns")
            st.write("- Trading recommendations based on analysis")
            
            pnl_by_sentiment = merged_data.groupby('classification')['total_pnl'].mean()
            insights = [
                f"Best performing sentiment: {pnl_by_sentiment.idxmax()} (Avg PnL: ${pnl_by_sentiment.max():,.2f})",
                f"Worst performing sentiment: {pnl_by_sentiment.idxmin()} (Avg PnL: ${pnl_by_sentiment.min():,.2f})",
                f"Overall win rate: {(merged_data['profitable_day'].mean()*100):.1f}%",
                f"Sentiment-PnL correlation: {merged_data['value'].corr(merged_data['total_pnl']):.3f}"
            ]
            
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report..."):
                    pdf_buffer = generate_pdf_report(merged_data, st.session_state['results'], insights)
                    st.session_state['pdf_buffer'] = pdf_buffer
                
                st.success("✅ PDF report generated successfully!")
            
            if 'pdf_buffer' in st.session_state:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state['pdf_buffer'],
                    file_name=f"sentiment_trading_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()

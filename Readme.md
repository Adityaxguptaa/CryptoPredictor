# Bitcoin Sentiment & Trader Performance ML Analytics

## Overview

This project is a comprehensive machine learning analytics platform that combines Bitcoin Fear & Greed Index sentiment data with cryptocurrency trader performance metrics from Hyperliquid. The application uses Streamlit to provide an interactive dashboard for analyzing the relationship between market sentiment and trading outcomes. It employs multiple traditional ML models (Random Forest, Gradient Boosting, Logistic Regression, XGBoost), deep learning models (LSTM, GRU), clustering algorithms, backtesting frameworks, and automated PDF reporting to deliver actionable trading insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Visualization Libraries**: 
  - Plotly (Express and Graph Objects) for interactive charts
  - Seaborn and Matplotlib for statistical visualizations
- **Layout**: Wide layout with expandable sidebar for navigation
- **Caching Strategy**: Uses `@st.cache_data` decorator to cache data loading and preprocessing operations, improving performance by preventing redundant computations

### Data Processing Pipeline
- **Input Sources**: Two CSV files containing fear/greed index data and historical trader data
- **Data Transformation**:
  - Date normalization across datasets
  - Aggregation of trader metrics by date (PnL, volume, trade counts, account statistics)
  - Feature engineering including buy/sell ratio calculation
  - Merge operation combining sentiment and trading data on date key
- **Design Rationale**: Daily aggregation enables time-series analysis and reduces noise from individual trades

### Machine Learning Architecture

**Traditional ML Models:**
- Random Forest Classifier (ensemble decision trees)
- Gradient Boosting Classifier (sequential ensemble)
- Logistic Regression (baseline linear model)
- XGBoost Classifier (optimized gradient boosting)

**Deep Learning Models:**
- LSTM (Long Short-Term Memory) - Captures long-term sequential dependencies
- GRU (Gated Recurrent Unit) - Efficient sequential pattern learning
- Both models use dropout layers, early stopping, and validation monitoring

**Clustering Algorithms:**
- K-Means clustering for trader segmentation
- DBSCAN for density-based clustering
- Silhouette score for cluster quality evaluation

**Backtesting Framework:**
- Sentiment-driven trading strategy simulation
- Multiple strategy types (extreme fear buy, sentiment momentum)
- Performance metrics: total return, Sharpe ratio, max drawdown, win rate

**Preprocessing & Evaluation:**
- StandardScaler for feature normalization
- Train/test split with stratification
- Comprehensive evaluation: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves
- Sequential data preparation for time-series models

**Design Rationale:** 
- Traditional ML for tabular feature relationships
- Deep learning for temporal pattern recognition
- Clustering for trader persona identification
- Backtesting for strategy validation

### Data Schema

**Fear & Greed Index Data**:
- `date`: Trading date
- Sentiment metrics (specific columns not shown in partial code)

**Trader Data**:
- `Timestamp IST`: Trade timestamp in Indian Standard Time
- `Closed PnL`: Profit and Loss for closed positions
- `Size USD`: Trade size in USD
- `Account`: Trader account identifier
- `Side`: Trade direction (BUY/SELL)

**Merged Daily Metrics**:
- `total_pnl`: Sum of all PnL for the day
- `avg_pnl`: Average PnL per trade
- `std_pnl`: Standard deviation of PnL (volatility measure)
- `num_trades`: Total number of trades
- `total_volume`: Aggregate trading volume
- `avg_trade_size`: Mean position size
- `unique_accounts`: Count of distinct traders
- `buy_percentage`: Percentage of buy-side trades

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework for interactive dashboards
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization library
- **seaborn/matplotlib**: Statistical plotting
- **scikit-learn**: Machine learning algorithms, preprocessing, evaluation metrics, clustering
- **xgboost**: Gradient boosting framework
- **tensorflow/keras**: Deep learning framework for LSTM and GRU models
- **scipy**: Scientific computing and statistical functions
- **reportlab**: PDF generation for automated reports

### Data Files
- `attached_assets/fear_greed_index_1761747110179.csv`: Bitcoin sentiment indicator data
- `attached_assets/historical_data_1761747110176.csv`: Cryptocurrency trading history

### Technical Considerations
- No database backend; relies on CSV file imports
- No authentication/authorization system implemented
- Application designed for local or single-user deployment
- Statistical analysis assumes daily aggregation is sufficient for pattern detection
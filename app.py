import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="INFY Stock Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Trend Prediction Dashboard - INFY")
st.write("This dashboard predicts whether INFY stock will go UP or DOWN the next day.")

# ── Load and prepare data ──────────────────────────────────────────────────────

df = pd.read_csv("INFY.NS.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Add technical indicators
df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()

# RSI calculation
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = delta.clip(upper=0).abs().rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

# MACD
df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()

# Volatility
df["Volatility"] = df["Close"].rolling(10).std()
df["Price_Change"] = df["Close"].pct_change()

df.dropna(inplace=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.header("Filter Options")
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date   = st.sidebar.date_input("End Date",   df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

st.sidebar.markdown("---")
st.sidebar.write(f"Total records: {len(filtered_df)}")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Charts", "📉 Indicators", "🤖 Model Results", "🔮 Predict"])

# ═══════════════════════════════════════════════════════
# TAB 1 - Data and Charts
# ═══════════════════════════════════════════════════════
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df[["Date","Open","High","Low","Close","Volume"]].tail(10))

    st.subheader("Basic Statistics")
    st.dataframe(filtered_df[["Open","High","Low","Close","Volume"]].describe().round(2))

    # Closing price chart
    st.subheader("Closing Price Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(filtered_df["Date"], filtered_df["Close"], color="steelblue", linewidth=1.2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (₹)")
    ax1.set_title("INFY Closing Price")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # Moving averages
    st.subheader("Moving Averages (MA10 and MA20)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(filtered_df["Date"], filtered_df["Close"], label="Close", color="steelblue", linewidth=1)
    ax2.plot(filtered_df["Date"], filtered_df["MA10"],  label="MA10",  color="orange",    linewidth=1.2)
    ax2.plot(filtered_df["Date"], filtered_df["MA20"],  label="MA20",  color="red",       linewidth=1.2)
    ax2.legend()
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (₹)")
    ax2.set_title("INFY with Moving Averages")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Volume
    st.subheader("Trading Volume")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.bar(filtered_df["Date"], filtered_df["Volume"], color="teal", alpha=0.6, width=1.5)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Volume")
    ax3.set_title("INFY Trading Volume")
    ax3.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Price distribution
    st.subheader("Closing Price Distribution")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.hist(filtered_df["Close"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax4.set_xlabel("Price (₹)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Closing Prices")
    ax4.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ═══════════════════════════════════════════════════════
# TAB 2 - Technical Indicators
# ═══════════════════════════════════════════════════════
with tab2:

    st.subheader("RSI - Relative Strength Index (14 days)")
    st.write("RSI above 70 = Overbought, RSI below 30 = Oversold")

    fig5, ax5 = plt.subplots(figsize=(10, 3))
    ax5.plot(filtered_df["Date"], filtered_df["RSI"], color="purple", linewidth=1.2)
    ax5.axhline(70, color="red",   linestyle="--", label="Overbought (70)")
    ax5.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    ax5.set_ylim(0, 100)
    ax5.legend()
    ax5.set_xlabel("Date")
    ax5.set_ylabel("RSI")
    ax5.set_title("RSI Indicator")
    ax5.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    latest_rsi = filtered_df["RSI"].iloc[-1]
    if latest_rsi > 70:
        st.warning(f"Current RSI is {latest_rsi:.1f} — Stock may be overbought")
    elif latest_rsi < 30:
        st.success(f"Current RSI is {latest_rsi:.1f} — Stock may be oversold")
    else:
        st.info(f"Current RSI is {latest_rsi:.1f} — Neutral zone")

    st.subheader("MACD Indicator")
    fig6, ax6 = plt.subplots(figsize=(10, 3))
    ax6.plot(filtered_df["Date"], filtered_df["MACD"], color="blue", linewidth=1.2, label="MACD")
    ax6.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax6.legend()
    ax6.set_xlabel("Date")
    ax6.set_ylabel("MACD Value")
    ax6.set_title("MACD")
    ax6.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    st.subheader("Price Volatility (10-day Rolling Std)")
    fig7, ax7 = plt.subplots(figsize=(10, 3))
    ax7.plot(filtered_df["Date"], filtered_df["Volatility"], color="darkorange", linewidth=1.2)
    ax7.set_xlabel("Date")
    ax7.set_ylabel("Volatility")
    ax7.set_title("Rolling Volatility (10 days)")
    ax7.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    st.subheader("Correlation Heatmap")
    cols = ["Open","High","Low","Close","Volume","MA10","MA20","RSI","MACD","Volatility"]
    fig8, ax8 = plt.subplots(figsize=(9, 6))
    sns.heatmap(filtered_df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax8)
    ax8.set_title("Feature Correlation")
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

# ═══════════════════════════════════════════════════════
# TAB 3 - Model Training and Results
# ═══════════════════════════════════════════════════════
with tab3:

    st.subheader("Model Training - Random Forest Classifier")
    st.write("We train a Random Forest model to predict if the stock will go UP (1) or DOWN (0) next day.")

    # Prepare features and target
    features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA20", "RSI", "MACD", "Volatility"]

    df2 = df.copy()
    df2["Target"] = (df2["Close"].shift(-1) > df2["Close"]).astype(int)
    df2.dropna(inplace=True)

    X = df2[features]
    y = df2["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save for prediction tab
    st.session_state["model"]   = model
    st.session_state["scaler"]  = scaler
    st.session_state["features"] = features

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",    f"{acc*100:.2f}%")
    col2.metric("Train Size",  f"{len(X_train)} samples")
    col3.metric("Test Size",   f"{len(X_test)} samples")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig9, ax9 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred DOWN","Pred UP"],
                yticklabels=["Actual DOWN","Actual UP"], ax=ax9)
    ax9.set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=["DOWN","UP"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("Feature Importance")
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig10, ax10 = plt.subplots(figsize=(7, 5))
    fi.plot(kind="barh", ax=ax10, color="steelblue")
    ax10.set_title("Feature Importance - Random Forest")
    ax10.set_xlabel("Importance Score")
    ax10.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close()

# ═══════════════════════════════════════════════════════
# TAB 4 - Prediction
# ═══════════════════════════════════════════════════════
with tab4:

    st.subheader("Predict Next Day Trend")
    st.write("Enter today's stock values to predict if INFY will go UP or DOWN tomorrow.")

    # Get last row values as defaults
    last = df.iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        open_val  = st.number_input("Open Price",  value=float(round(last["Open"],  2)))
        high_val  = st.number_input("High Price",  value=float(round(last["High"],  2)))
        low_val   = st.number_input("Low Price",   value=float(round(last["Low"],   2)))
        close_val = st.number_input("Close Price", value=float(round(last["Close"], 2)))
        vol_val   = st.number_input("Volume",      value=float(last["Volume"]))

    with col2:
        ma10_val  = st.number_input("MA10",       value=float(round(last["MA10"],  2)))
        ma20_val  = st.number_input("MA20",       value=float(round(last["MA20"],  2)))
        rsi_val   = st.number_input("RSI",        value=float(round(last["RSI"],   2)))
        macd_val  = st.number_input("MACD",       value=float(round(last["MACD"],  4)))
        volat_val = st.number_input("Volatility", value=float(round(last["Volatility"], 4)))

    if st.button("Predict"):

        if "model" not in st.session_state:
            st.warning("Please go to the Model Results tab first to train the model.")
        else:
            input_data = [[
                open_val, high_val, low_val, close_val, vol_val,
                ma10_val, ma20_val, rsi_val, macd_val, volat_val
            ]]

            input_scaled = st.session_state["scaler"].transform(input_data)
            prediction   = st.session_state["model"].predict(input_scaled)[0]
            probability  = st.session_state["model"].predict_proba(input_scaled)[0]

            st.markdown("---")
            if prediction == 1:
                st.success(f"📈 Stock is likely to go UP tomorrow  (Confidence: {probability[1]*100:.1f}%)")
            else:
                st.error(f"📉 Stock is likely to go DOWN tomorrow  (Confidence: {probability[0]*100:.1f}%)")

            # Show probability bar
            fig11, ax11 = plt.subplots(figsize=(5, 3))
            ax11.bar(["DOWN", "UP"], probability, color=["salmon", "mediumseagreen"], width=0.4)
            ax11.set_ylim(0, 1)
            ax11.set_ylabel("Probability")
            ax11.set_title("Prediction Confidence")
            for i, v in enumerate(probability):
                ax11.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=12)
            ax11.grid(alpha=0.2, axis="y")
            plt.tight_layout()
            st.pyplot(fig11)
            plt.close()

            st.caption("⚠️ Note: This is a machine learning prediction for educational purposes only. Do not use it for actual investment decisions.")

import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go #type:ignore

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Netflix data.csv')
    return df


df = load_data()

st.set_page_config(layout="wide")
         
#st.title("Netflix Data Dashboard")

# -----------------------------
# Data Types
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

# Use release_year as time axis (force numeric if needed)
if 'release_year' in df.columns:
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Controls")

# Histogram controls
st.sidebar.subheader("Histogram")
hist_col = st.sidebar.selectbox("Variable", numeric_cols, key="hist", index=numeric_cols.index("imdb_score"))
min_val = float(df[hist_col].min())
max_val = float(df[hist_col].max())
hist_range = st.sidebar.slider(
            "Histogram Range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
            )

# Time series controls
st.sidebar.subheader("Time Series")
valid_ts_cols = [col for col in numeric_cols if col != 'release_year']  
ts_cols = st.sidebar.multiselect(
    "Variables",
    valid_ts_cols,
    max_selections=2,
    key="ts",
    default=["imdb_score", "runtime"]
)
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Bar chart controls
st.sidebar.subheader("Bar Chart")
bar_cols = st.sidebar.selectbox("Variables", numeric_cols, key="bar", index=numeric_cols.index("imdb_score"))

# Scatter controls
st.sidebar.subheader("Scatter Plot")
x_col = st.sidebar.selectbox("X variable", numeric_cols, key="x", index=numeric_cols.index("imdb_votes"))
y_col = st.sidebar.selectbox("Y variable", numeric_cols, key="y", index=numeric_cols.index("imdb_score"))
x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
y_min, y_max = float(df[y_col].min()), float(df[y_col].max())

x_range = st.sidebar.slider(
    "X Range",
    min_value=x_min,
    max_value=x_max,
    value=(x_min, x_max)
)

y_range = st.sidebar.slider(
    "Y Range",
    min_value=y_min,
    max_value=y_max,
    value=(y_min, y_max)
)

# -----------------------------
# HISTOGRAM
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    if hist_col:
        filtered_df = df[(df[hist_col] >= hist_range[0]) & (df[hist_col] <= hist_range[1])]

        fig = px.histogram(filtered_df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}", color_discrete_sequence=["darkred"])
        fig.update_traces(
            marker=dict(
                line=dict(
                    color="black",  # border color
                    width=1         # thickness
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TIME SERIES
# -----------------------------
with col2:
    if 'release_year' in df.columns and len(ts_cols) > 0:

        ts_df = df[
            (df['release_year'] >= year_range[0]) &
            (df['release_year'] <= year_range[1])
        ]

        # aggregation
        ts_cols_clean = [col for col in ts_cols if col != 'release_year']

        ts_grouped = (
            ts_df.groupby('release_year')[ts_cols_clean]
            .mean()
            .reset_index()
        )
        fig = go.Figure()

        # --- First variable (LEFT axis, RED) ---
        fig.add_trace(
            go.Scatter(
                x=ts_grouped['release_year'],
                y=ts_grouped[ts_cols[0]],
                name=ts_cols[0],
                line=dict(color='darkred', width=3),
                yaxis='y1'
            )
        )

        # --- Second variable (RIGHT axis, BLUE) ---
        if len(ts_cols) == 2:
            fig.add_trace(
                go.Scatter(
                    x=ts_grouped['release_year'],
                    y=ts_grouped[ts_cols[1]],
                    name=ts_cols[1],
                    line=dict(color='darkblue', width=3),
                    yaxis='y2'
                )
            )

        # layout with dual axis
        fig.update_layout(
            title="Average over Time",
            xaxis=dict(title="Release Year"),

            yaxis=dict(
                title=dict(
                    text=ts_cols[0],
                    font=dict(color="white")
                ),
                tickfont=dict(color="white")
            ),

            yaxis2=dict(
                title=dict(
                    text=ts_cols[1] if len(ts_cols) == 2 else "",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white"),
                overlaying='y',
                side='right',
                showgrid=False
            ),

            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
# -----------------------------
# BAR CHART
# -----------------------------
with col1:
    if bar_cols:
        sentiment_col = 'Bert_class'
        bar_df = (
            df.groupby(sentiment_col)[bar_cols]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            bar_df,
            x=sentiment_col,
            y=bar_cols,
            title=f"Average {bar_cols} by Sentiment",
            color_discrete_sequence=["darkred"],
            labels={sentiment_col: "Sentiment"}
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SCATTER PLOT
# -----------------------------
with col2:
    if x_col and y_col:
        scatter_df = df[
            (df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1]) &
            (df[y_col] >= y_range[0]) & (df[y_col] <= y_range[1])
        ]

        fig = px.scatter(scatter_df, x=x_col, y=y_col, title='Scatter Plot', color_discrete_sequence=["darkred"])
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go 


def plot_histogram():
    if hist_col:
        filtered_df = df_filtered[(df_filtered[hist_col] >= hist_range[0]) & (df_filtered[hist_col] <= hist_range[1])]

        fig = px.histogram(filtered_df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}", color_discrete_sequence=[NETFLIX_COLOR])
        fig.update_traces(
            marker=dict(
                line=dict(
                    color="black",  # border color
                    width=1         # thickness
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_time_series():
    if 'release_year' in df.columns and len(ts_cols) > 0:

            ts_df = df_filtered[
                (df_filtered['release_year'] >= year_range[0]) &
                (df_filtered['release_year'] <= year_range[1])
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
                    line=dict(color=NETFLIX_COLOR, width=3),
                    yaxis='y1'
                )
            )

            # --- Second variable (RIGHT axis, teal) ---
            if len(ts_cols) == 2:
                fig.add_trace(
                    go.Scatter(
                        x=ts_grouped['release_year'],
                        y=ts_grouped[ts_cols[1]],
                        name=ts_cols[1],
                        line=dict(color='teal', width=3),
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

def plot_bar_chart(x_bar):
    if bar_cols:
            bar_df = (
                df_filtered.groupby(x_bar)[bar_cols]
                .mean()
                .reset_index()
            )

            min_val = bar_df[bar_cols].min()
            max_val = bar_df[bar_cols].max()
            pad = (max_val - min_val) * 0.10
            range_y = [min_val - pad, max_val + pad]

            fig = px.bar(
                bar_df,
                x=x_bar,
                y=bar_cols,
                title=f"Average {bar_cols} by {x_bar}",
                color_discrete_sequence=[NETFLIX_COLOR],
                labels={x_bar: x_bar},
                range_y=range_y
            )

            st.plotly_chart(fig, use_container_width=True)

def plot_scatter():
    if x_col and y_col:
        scatter_df = df_filtered[
            (df_filtered[x_col] >= x_range[0]) & (df_filtered[x_col] <= x_range[1]) &
            (df_filtered[y_col] >= y_range[0]) & (df_filtered[y_col] <= y_range[1])
        ]
    ts_cols_clean = [col for col in ts_cols if col != 'release_year']
    scatter_df[ts_cols_clean] = scatter_df[ts_cols_clean].round(2)
    fig = px.scatter(
        scatter_df, 
        x=x_col,
        y=y_col,
        title='Scatter Plot', 
        color_discrete_sequence=[NETFLIX_COLOR],
        opacity=0.6,
        hover_name='title',
        hover_data={
            "release_year": True,
            "imdb_score": True,
            "imdb_votes": True,
            "Bert_class": True
        }
        )
    fig.update_yaxes(tickformat=".2s")
    fig.update_xaxes(tickformat=".2s")

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_data():
    df = pd.read_csv('Netflix data.csv')
    return df


df = load_data()

st.set_page_config(layout="wide")
NETFLIX_COLOR = "#E50914"        

# Data Types
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()


# SIDEBAR CONTROLS
st.sidebar.header("Controls")

type_option = st.sidebar.radio(
    "Select Content Type",
    options=["All", "Movie", "Show"],
    index=0,
    horizontal=True,
)
# Apply type filter
if type_option == "All":
    df_filtered = df.copy()
else:
    mapping = {"Movie": "MOVIE", "Show": "SHOW"}
    df_filtered = df[df["type"] == mapping[type_option]]

# Histogram controls
st.sidebar.subheader("Histogram")
hist_col = st.sidebar.selectbox("Variable", numeric_cols, key="hist", index=numeric_cols.index("imdb_score"))
min_val = float(df_filtered[hist_col].min())
max_val = float(df_filtered[hist_col].max())
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
min_year = int(df_filtered['release_year'].min())
max_year = int(df_filtered['release_year'].max())

year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Bar chart controls
st.sidebar.subheader("Bar Chart")
sentiment_or_age_rating = st.sidebar.radio(
    "Select Type",
    options=["Sentiment", "Age certification"],
    index=0,
    horizontal=True
)
x_bar_mapping = {"Sentiment": "Bert_class", "Age certification": "age_certification"}
x_bar = x_bar_mapping.get(sentiment_or_age_rating)
valid_options = [col for col in numeric_cols if col != "release_year"]
bar_cols = st.sidebar.selectbox("Variable", valid_options, key="bar", index=valid_options.index("imdb_score"))
# Scatter controls
st.sidebar.subheader("Scatter Plot")
x_col = st.sidebar.selectbox("X variable", numeric_cols, key="x", index=numeric_cols.index("imdb_votes"))
valid_y_options = [col for col in numeric_cols if col != x_col]
default_y = "imdb_score"
default_y = valid_y_options[0] if default_y not in valid_y_options else default_y
y_col = st.sidebar.selectbox("Y variable", valid_y_options, key="y", index=valid_y_options.index(default_y))
x_min, x_max = float(df_filtered[x_col].min()), float(df_filtered[x_col].max())
y_min, y_max = float(df_filtered[y_col].min()), float(df_filtered[y_col].max())

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

col1, col2 = st.columns(2)
with col1:
    plot_histogram()
    plot_bar_chart(x_bar)

with col2:
    plot_time_series()
    plot_scatter()



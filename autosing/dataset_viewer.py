import sys
import pathlib

import streamlit as st
import pandas as pd

from alignment import Alignment

st.set_page_config(layout="wide")

partition = pathlib.Path(sys.argv[1])
index = pd.read_parquet(partition / "chunked.parquet").set_index("id")
vocals = pd.read_parquet(partition / "vocals.parquet").set_index("id")
no_vocals = pd.read_parquet(partition / "no_vocals.parquet").set_index("id")
stoks = pd.read_parquet(partition / "stoks.parquet").set_index("id")

vocals = vocals.rename(columns={"audio": "vocals"})
no_vocals = no_vocals.rename(columns={"audio": "no_vocals"})

df = pd.concat([index, vocals, no_vocals, stoks], axis=1, join="inner").sample(frac=1)

st.title("Dataset Viewer")
st.markdown("---")

# Loop through the DataFrame to display audio and corresponding text
for _, (idx, row) in zip(range(20), df.iterrows()):
    if row["alignment"] is not None:
        alignment = Alignment(**row["alignment"])
    else:
        alignment = None

    with st.container():
        cols = st.columns([1, 6])

        if alignment is not None:
            with cols[0]:
                st.markdown("#### Inlier Ratio")
                st.write(round(alignment.inlier_ratio, 2))
            with cols[1]:
                st.markdown("#### Lyrics")
                st.write(" ".join(alignment.words))
        else:
            st.write("Missing Lyrics")

    with st.container():
        cols = st.columns(3)
        with cols[0]:
            st.markdown("#### Full Track")
            st.audio(row['audio'], format='audio/mp3')
        with cols[1]:
            st.markdown("#### Vocals")
            st.audio(row['vocals'], format='audio/mp3')
        with cols[2]:
            st.markdown("#### Instrumentals")
            st.audio(row['no_vocals'], format='audio/mp3')


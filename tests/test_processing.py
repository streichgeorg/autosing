import os
import subprocess

import pandas as pd
from pathlib import Path

import pytest

num_samples = 4
num_partitions = 2

def execute_cmd(cmd, cli_args=[]):
    subprocess.check_output([
        "python", "autosing/processing.py",
        "--num", str(num_samples),
        "--num-partitions", str(num_partitions),
        "--batch-size", "2",
        os.environ["AUTOSING_DATASET"], cmd, "0", *cli_args
    ])

def load_df(name, dataset_dir=Path(os.environ["AUTOSING_DATASET"]), partition=0):
    return pd.read_parquet(dataset_dir / str(partition) / f"{name}.parquet")

def correct_num_samples(name, num=num_samples):
    for i in range(num_partitions):
        df = load_df(name, partition=i)
        assert len(df) == num

def test_vad_extract():
    execute_cmd("vad_extract")
    correct_num_samples("chunked")

def test_align():
    execute_cmd("align")
    correct_num_samples("alignment")

def test_src_sep():
    execute_cmd("src_sep")
    correct_num_samples("vocals")
    correct_num_samples("no_vocals")

def test_atoks():
    execute_cmd("atoks", ["--audio-src", "vocals"])
    correct_num_samples("atoks_vocals_snac_32khz")
    correct_num_samples("atoks_no_vocals_snac_32khz")

def test_stoks():
    execute_cmd("stoks")
    correct_num_samples("stoks")

def test_artist_embs():
    execute_cmd("artist_embs")
    correct_num_samples("artist_embs")


import argparse
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from autosing.codecs import load_codec_model
from autosing.alignment import Alignment

def load_df(path, columns=None):
    df = pd.read_parquet(path, columns=columns)
    df = df.set_index("id")
    return df.loc[~df.index.duplicated(keep='first')]

codec = load_codec_model("snac_32khz")

def split_partition(df, right):
    df = df.copy()

    frame_length = int(15 * codec.frame_rate) // codec.alignment * codec.alignment

    def split_atoks(atoks):
        atoks = atoks.apply(lambda x: x.reshape(4, -1))
        offset = right * frame_length
        atoks = atoks.apply(lambda x: x[:, offset:offset+frame_length].reshape(-1))
        return atoks

    df["atoks"] = split_atoks(df["atoks"])

    if "mtoks" in df.columns:
        df["mtoks"] = split_atoks(df["mtoks"])

    offset = right * 375
    df["stoks"] = df["stoks"].apply(lambda x: x[offset:offset + 375])
    df["embs"] = df["embs"].apply(lambda x: x.reshape(3, -1)[2 * right])

    if "alignment" in df.columns:
        alignment = df["alignment"].apply(
            lambda x: Alignment(**x).slice(15 * right, 15 * (right + 1))
        )
        df = df[alignment.apply(lambda x: x.cover()) / 15 > 0.6]

    return df
    
def load_partition(dataset_dir, idx, split, needs_mtoks=True):
    dataset_dir = Path(dataset_dir)
    try:
        path = dataset_dir / str(idx)

        atoks = load_df(path / "atoks_vocals_snac_32khz.parquet")
        atoks = atoks.rename(columns={"values": "atoks"})

        mtoks = []
        if needs_mtoks:
            mtoks = load_df(path / "atoks_no_vocals_snac_32khz.parquet")
            mtoks = [mtoks.rename(columns={"values": "mtoks"})]

        stoks = load_df(path / "stoks.parquet")
        stoks = stoks.rename(columns={"values": "stoks"})

        embs = load_df(path / "artist_embs.parquet")

        alignment = load_df(
            path / "chunked.parquet",
            columns=["id", "alignment"]
        )

        df = pd.concat([atoks, *mtoks, stoks, embs, alignment], axis=1, join="inner")

        if split:
            df = pd.concat([split_partition(df, False), split_partition(df, True)])

        return df
    except Exception:
        print(f"{dataset_dir} - {idx}:")
        print(traceback.format_exc())

def write_df(name, df, output_dir, num_partitions):
    df = df.sample(frac=1).reset_index(drop=True)

    chunk_size = len(df) // num_partitions
    for i in tqdm(range(num_partitions)):
        part_dir = output_dir / str(i)
        part_dir.mkdir(exist_ok=True, parents=True)

        partition_df = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        partition_df.to_parquet(part_dir / f"{name}.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="The location of the original data")
    parser.add_argument("num_in_partitions", type=int, help="Number of partitions the original data was split into")
    parser.add_argument("output_dir", type=Path, help="Output directory for the dataset")
    parser.add_argument("--num-out-partitions", type=int, default=512, help="Number of partitions that the dataset will be split into. This needs to be large enough so that every worker has its own partition during training. E.g. when train across 64 GPUS with 8 dataloader workers per GPU at least 512 partitions are required.")
    parser.add_argument("--split", type=bool, default=False, help="Whether to split the data into half (e.g. 15 second chunks). Useful to perform experiments.")
    parser.add_argument("--name", type=str, default="main")

    args = parser.parse_args()

    write_df(
        args.name,
        pd.concat([
            load_partition(args.data_dir, i, args.split)
            for i in tqdm(range(args.num_in_partitions))
        ]),
        args.output_dir,
        # Create an extra partition for the validation set
        args.num_out_partitions + 1,
    )


import argparse
import dataclasses
import io
import pathlib
import sys
import traceback

import pandas as pd
from tqdm import tqdm


help_text = """
The individual processing steps have the following dependencies:

    align
      |
      v
    vad_extract
      |
      v
    src_sep
      /  \\
     v    v
    atoks  stoks
      |
      v
    artist_embs

The atoks step will have to be executed for both the vocals and no_vocals track.
"""

def align(args):
    from autosing.alignment import compute_alignment, AlignException

    if args.fc_model is None:
        raise Exception(
            "Please pass the path to the phoneme classification model you want to use."
        )

    dataset_dir = pathlib.Path(args.dir)

    table = pd.read_parquet(
        dataset_dir / f"{args.partition}/index.parquet"
    ).set_index("id")
    it = table.to_dict(orient="records")

    df = pd.DataFrame(columns=["id", "alignment"])

    for row_id, row in zip(table.index, tqdm(it, file=sys.stdout)):
        try:
            if "lrc_lyrics" in row:
                alignment, _ = compute_alignment(
                    io.BytesIO(row["audio"]),
                    lrc_text=row.get("lrc_lyrics", None),
                    model_name=args.fc_model,
                )
                alignment = dataclasses.asdict(alignment)
            elif "raw_lyrics" in row:
                _, alignment = compute_alignment(
                    io.BytesIO(row["audio"]),
                    raw_text=row.get("raw_lyrics", None),
                    model_name=args.fc_model,
                )
                alignment = dataclasses.asdict(alignment)
            else:
                alignment = None
        except AlignException:
            traceback.format_exc()
            alignment = None

        df.loc[len(df)] = [row_id, alignment]

        if len(df) >= args.num: break

    df.to_parquet(dataset_dir / f"{args.partition}/alignment.parquet")

def interesting_chunks(chunk_length, mask, occupancy=None):

    num_seconds = mask.shape[-1]

    def score_function(ratio):
        if ratio < occupancy: return 0
        else: return 1

    # I think we don't actually need DP here but previously I experimented with
    # some other score functions where it was useful
    T = [0 for _ in range(num_seconds + 1)]

    for i in range(chunk_length, num_seconds + 1):
        chunk_score = score_function(mask[i - chunk_length:i].sum() / chunk_length)
        T[i] = max(T[i - chunk_length] + chunk_score, T[i - 1])

    i = num_seconds
    offsets = []

    while i > chunk_length:
        if T[i] == T[i - 1]:
            i -= 1
        else:
            i -= chunk_length
            offsets.append(i)

    return offsets

def vad_extract(args):
    import torchaudio
    from autosing.alignment import compute_va, Alignment

    dataset_dir = pathlib.Path(args.dir)
    index = pd.read_parquet(
        dataset_dir / f"{args.partition}"/ "index.parquet"
    ).set_index("id")

    alignment_file = dataset_dir / f"{args.partition}"/ "alignment.parquet"
    if alignment_file.exists():
        alignment = pd.read_parquet(alignment_file).set_index("id")
        index = pd.concat([index, alignment], axis=1, join="inner")

    df = pd.DataFrame(columns=["id", "audio", "alignment"])

    it = zip(
        index.index,
        tqdm(index.to_dict(orient="records"), file=sys.stdout),
        range(args.num)
    )

    for row_id, row, _ in it:
        try:
            va = compute_va(
                io.BytesIO(row["audio"]),
                model_name="streich/singing_va",
            ) 
        except Exception:
            print(traceback.format_exc())

        offsets = interesting_chunks(args.chunk_length, va, args.min_occupancy)

        if len(offsets) == 0: continue

        audio, sample_rate = torchaudio.load(io.BytesIO(row["audio"]))
        for offset in offsets:
            chunk = audio[:, offset * sample_rate:(offset + args.chunk_length) * sample_rate]
            buf = io.BytesIO()
            torchaudio.save(buf, chunk, sample_rate, format=args.audio_format)
            buf.seek(0)

            if "alignment" in row and row["alignment"] is not None:
                alignment = Alignment(**row["alignment"]).slice(offset, offset + args.chunk_length)
                alignment = dataclasses.asdict(alignment)
            else:
                alignment = None

            df.loc[len(df)] = [
                f"{row_id}_at_{offset}",
                buf.read(),
                alignment,
            ]

    df.to_parquet(dataset_dir / f"{args.partition}" / "chunked.parquet")

def src_sep(args):
    import torch
    import torchaudio
    from autosing.simple_dataset import AudioDataset
    import torch.multiprocessing as mp
    from audio_separator.separator import Separator

    separator = Separator(
        model_file_dir=pathlib.Path.home() / ".cache/audio-separator-models"
    )
    separator.load_model()
    ds = AudioDataset(args.dir, [args.partition], ["chunked"], sample_rate=separator.sample_rate)

    dataset_dir = pathlib.Path(args.dir)

    def save_to_bytes(x):
        x = torch.tensor(x)
        buf = io.BytesIO()
        torchaudio.save(buf, x, sample_rate=separator.sample_rate, format="mp3")
        buf.seek(0)
        return buf.read()

    def encoder(dataset_dir, name, queue):
        df = pd.DataFrame(columns=["id", "audio"])
        for ids, stem in iter(queue.get, None):
            for eid, audio in zip(ids, stem):
                df.loc[len(df)] = [eid, save_to_bytes(audio)]
        df.to_parquet(dataset_dir / str(args.partition) / f"{name}.parquet")

    manager = mp.Manager()

    vocal_queue = manager.Queue(10)
    vocal_encoder = mp.Process(target=encoder, args=(dataset_dir, "vocals", vocal_queue))

    no_vocal_queue = manager.Queue(10)
    no_vocal_encoder = mp.Process(target=encoder, args=(dataset_dir, "no_vocals", no_vocal_queue))

    vocal_encoder.start()
    no_vocal_encoder.start()

    count = 0
    for ids, stems in separator.separate_batched(ds, args.batch_size):
        vocal_queue.put((ids, stems["Vocals"]))
        no_vocal_queue.put((ids, stems["Instrumental"]))
        count += len(ids)
        if count > args.num: break

    vocal_queue.put(None)
    no_vocal_queue.put(None)

    vocal_encoder.join()
    no_vocal_encoder.join()

def atoks(args):
    import torch

    from autosing.simple_dataset import AudioDataset
    from autosing.codecs import load_codec_model

    codec = load_codec_model(args.codec)

    ds = AudioDataset(
        args.dir,
        [args.partition],
        [args.audio_src],
        sample_rate=codec.sample_rate,
        num_channels=1,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=4)

    result = pd.DataFrame(columns=["id", "values"])

    for entry in tqdm(dl):
        batched_atoks = codec.encode(entry["audio"]).cpu()
        for eid, atoks in zip(entry["id"], batched_atoks):
            result.loc[len(result)] = [eid, atoks.reshape(-1).numpy()]

    dataset_dir = pathlib.Path(args.dir)
    result.to_parquet(
        dataset_dir / str(args.partition) / f"atoks_{args.audio_src}_{args.codec}.parquet"
    )

def stoks(args):
    import torch
    from einops import rearrange

    from autosing.simple_dataset import AudioDataset
    from autosing.bn_whisper import load_model

    model = load_model("streich/bn_whisper").cuda()

    sample_rate = 16000
    ds = AudioDataset(
        args.dir,
        [args.partition],
        ["vocals"],
        sample_rate=sample_rate,
        num_channels=1,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=8)

    result = pd.DataFrame(columns=["id", "values"])

    for i, entry in enumerate(tqdm(dl)):
        audio = rearrange(entry["audio"][:, :, :30 * sample_rate], "b 1 (n t) -> (b n) t", n=2)
        with torch.inference_mode():
            batched_stoks = model.encode_audio(audio)
        batched_stoks = rearrange(batched_stoks, "(b n) t -> b (n t)", n=2)

        for j, (eid, stoks) in enumerate(zip(entry["id"], batched_stoks)):
            result.loc[len(result)] = [eid, stoks.cpu().numpy()]

    dataset_dir = pathlib.Path(args.dir)
    result.to_parquet(
        dataset_dir / str(args.partition) / "stoks.parquet"
    )

def artist_embs(args):
    import torch
    from einops import rearrange

    from autosing.artist_embeddings import load_model, EmbeddingDataset

    model = load_model("streich/artist_emb:artist_emb.model").cuda()

    ds = EmbeddingDataset(
        args.dir,
        [args.partition],
        inference=True,
        chunk_length=30,
        shufbuf_size=1,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=1)

    result = pd.DataFrame(columns=["id", "embs"])

    for i, entry in enumerate(tqdm(dl)):
        atoks = rearrange(entry["atoks"], "b q (n t) -> (b n) q t", n=3)

        with torch.inference_mode():
            batched_embs = model(atoks.cuda(), noloss=True).cpu()
        batched_embs = rearrange(batched_embs, "(b n) ... -> b n ...", n=3)

        for eid, embs in zip(entry["id"], batched_embs):
            result.loc[len(result)] = [eid, embs.cpu().numpy().reshape(-1)]

    dataset_dir = pathlib.Path(args.dir)
    result.to_parquet(
        dataset_dir / str(args.partition) / "artist_embs.parquet"
    )

def missing(dataset_dir, what, n):
    dataset_dir = pathlib.Path(dataset_dir)
    done = set(int(f.parts[-2]) for f in pathlib.Path(dataset_dir).glob(f"**/{what}"))
    missing = sorted(list(set(range(n)) - done))
    return missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=help_text)

    parser.add_argument("dir", help="Directory where the dataset is located.")
    parser.add_argument(
        "cmd",
        choices=["src_sep", "align", "vad_extract", "atoks", "stoks", "artist_embs"],
        help="Processing steps"
    )
    parser.add_argument("partition", type=int)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument(
        "--min-occupancy",
        type=float, default=0.8,
        help="Minimum singing fraction when extracting chunks"
    )
    parser.add_argument("--codec", default="snac_32khz")
    parser.add_argument(
        "--num", type=int, default=10000000,
        help="Limit numbers of rows to process (for debugging)"
    )
    parser.add_argument(
        "--chunk-length", type=int, default=31,
        help="Length of extracted chunks"
    )
    parser.add_argument("--audio-src", default="chunked")
    parser.add_argument("--fc-model", default="streich/singing_va")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--complete-missing", type=str, default=None)
    parser.add_argument("--total-partitions", type=int, default=None)
    parser.add_argument("--audio-format", type=str, default="opus")

    args = parser.parse_args()
    print(args)

    if args.complete_missing is not None:
        missing_partitions = missing(
            args.dir, args.complete_missing, args.total_partitions
        )

    partition = args.partition
    for i in range(args.num_partitions):
        if args.complete_missing is not None:
            if partition + i >= len(missing_partitions): break
            args.partition = missing_partitions[partition + i]
            print(f"Completing partition {args.partition}")
        else:
            args.partition = partition + i

        try:
            if args.cmd == "src_sep":
                src_sep(args)
            elif args.cmd == "align":
                align(args)
            elif args.cmd == "vad_extract":
                vad_extract(args)
            elif args.cmd == "atoks":
                atoks(args)
            elif args.cmd == "stoks":
                stoks(args)
            elif args.cmd == "artist_embs":
                artist_embs(args)
            else:
                print("Invalid cmd", file=sys.stderr)
        except Exception as e:
            if args.num_partitions == 1:
                raise e
            else:
                print(traceback.format_exc())
        

import sys
import pathlib
import io
import random
import traceback
from datetime import datetime
from random import Random

import pandas as pd
import torch
import torchaudio

class SimpleDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_dir,
        subset,
        select=None,
        shufbuf_size=1,
        shuffle_df=False,
        random_state=None,
    ):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.subset = subset
        assert select is None or len(select) == 1
        self.select = select
        self.shufbuf_size = shufbuf_size
        self.shuffle_df = shuffle_df

        self.random_state = random_state if random_state is not None else random.randint(0, 100000)

        self.ddp_rank = 0
        self.ddp_world_size = 1

    def shard_info(self):
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        rank = num_workers * self.ddp_rank + worker_id
        world_size = num_workers * self.ddp_world_size

        return rank, world_size

    def log(self, s):
        rank, world_size = self.shard_info()
        timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"shard {rank}, {world_size} - {timestr} - {s}", file=sys.stderr)

    def load_df(self, idx, name, columns=None):
        df = pd.read_parquet(
            self.dataset_dir / str(idx) / f"{name}.parquet",
            columns=columns,
            use_threads=False, pre_buffer=False, memory_map=True,
        )

        if "id" in df.columns:
            df = df.set_index("id")
            df = df.loc[~df.index.duplicated(keep='first')]

        return df

    def prepare_df(self, idx):
        df = self.load_df(idx, self.select[0])
        df["id"] = df.index
        return df

    def partitions(self):
        import pyarrow
        pyarrow.jemalloc_set_decay_ms(0)

        rank, world_size = self.shard_info()

        if len(self.subset) >= world_size:
            if len(self.subset) % world_size != 0:
                self.log("warning!!!, your dataset is not evenly divisible across workers")
            subset = self.subset[rank::world_size]
            shard_df = False
        else:
            subset = self.subset
            shard_df = True

        self.log(f"starting epoch {subset}")

        for idx in subset:
            df = self.prepare_df(idx)
            if self.shuffle_df: df = df.sample(
                frac=1, random_state=self.random_state + self.shard_info()[0]
            ).reset_index(drop=True)
            if shard_df:
                n = len(df) // world_size * world_size
                df = df.iloc[rank:n:world_size]

            yield df

    def shuffle(self, chunks):
        rand = Random(42)

        buf = []
        for _, entry in zip(range(self.shufbuf_size), chunks):
            buf.append(entry)
        rand.shuffle(buf)

        for entry in chunks:
            idx = rand.randint(0, self.shufbuf_size - 1)
            yield buf[idx]
            buf[idx] = entry

        yield from buf
        rand.shuffle(buf)
        yield from buf

    def entries(self):
        for df in self.partitions():
            yield from df.to_dict(orient="records")

    def __iter__(self):
        try:
            it = self.entries()
            if self.shufbuf_size > 1:
                yield from self.shuffle(it)
            else:
                yield from it
        except Exception as e:
            rank, world_size = self.shard_info()
            print("shard {rank}/{world_size} failed", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            raise e

    @property
    def total_samples(self):
        if not hasattr(self, "total_samples_"):
            df = next(self.partitions())
            self.total_samples_ = len(self.subset) * len(df)
        return self.total_samples_

    @property
    def total_length(self):
        return 30 * self.total_samples

    def __len__(self):
        return self.total_samples

class AudioDataset(SimpleDataset):
    def __init__(
        self,
        *args,
        sample_rate=None,
        num_channels=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        assert num_channels != 2
        self.num_channels = num_channels

    def entries(self):

        for entry in super().entries():
            audio, sample_rate = torchaudio.load(io.BytesIO(entry["audio"]))
            if self.sample_rate is not None:
                audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
                sample_rate = self.sample_rate

            if self.num_channels is not None:
                audio = audio[:self.num_channels]

            yield {
                **entry,
                "audio": audio,
                "sample_rate": sample_rate,
            }


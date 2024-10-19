import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastcore.basics import store_attr
from huggingface_hub import hf_hub_download

from whisperspeech.s2a_delar_mup_wds_mlang import DelSumEmbedding, DelSumHead
from whisperspeech.modules import *

from autosing.simple_dataset import SimpleDataset
from autosing.codecs import flatten_snac, unflatten_snac

@dataclasses.dataclass
class Tunables:
    init_std :float = 9
    embeddings_std :float = 0.2
    embeddings_lr_scale: float = 10
    output_mult :float = 5.6
    # FIXME: try separate mults for self and cross attention
    query_mult :float = .3
    linear_heads :bool = False
    rope :bool = True

    class_loss_ratio: float = 0.3
    label_smoothing: float = 0.1
    
    lr0 :float = 1e-3
    clip_gradient_norm :float = 2
    weight_decay :float = 1e-3
    warmup_steps :float = 500

class Embedder(nn.Module):
    def __init__(
        self,
        num_classes,
        ctx_n,
        depth=8, n_head=12, head_width=64, ffn_mult=4, emb_width=256,
        quantizers=4, codes=4096, tunables=Tunables(),
        finetune=False,
    ):
        super().__init__()

        self.codes = codes
        width = n_head * head_width

        store_attr("depth,ctx_n,codes,n_head,head_width,ffn_mult,quantizers,num_classes,emb_width,finetune")

        self.width = width
        self.base_width = 3 * head_width
        self.tunables = tunables

        qk_scale = self.tunables.query_mult * 8 / math.sqrt(head_width)

        self.encoder = nn.Sequential(*[
            ResidualAttentionBlock(width, n_head, qk_scale=qk_scale, ffn_mult=ffn_mult, rope=tunables.rope) for _ in range(depth)
        ])

        self.embds = DelSumEmbedding(
            pos_embs=None, length=ctx_n,
            n_head=n_head, head_width=head_width, atoks_width=None,
            quantizers=quantizers, codes=self.codes,
        )

        self.emb_proj = nn.Linear(self.width, emb_width)
        self.class_head = nn.Linear(emb_width, num_classes)
        self.ar_head = DelSumHead(n_head=n_head, head_width=head_width, quantizers=quantizers)

        if finetune:
            def freeze(m): m.lr_scale = 0
            self.embds.apply(freeze)
            self.encoder.apply(freeze)
            self.emb_proj.apply(freeze)
            self.ar_head.apply(freeze)
        else:
            self.apply(self.init_transformer)

    def init_transformer(self, m):
        if isinstance(m, LinearHead):
            m.no_weight_decay = True
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, QueryHead):
            m.lr_scale = 1/(m.weight.shape[1] / self.base_width)
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, nn.Embedding):
            m.no_weight_decay = True
            m.lr_scale = self.tunables.embeddings_lr_scale
            std = self.tunables.embeddings_std
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-3*std, b=3*std)
        elif isinstance(m, nn.Linear):
            m.lr_scale = 1/(m.weight.shape[1] / self.base_width)
            std = self.tunables.init_std / m.weight.shape[1]
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-3*std, b=3*std)
            if m.bias is not None:
                torch.nn.init.trunc_normal_(m.bias, std=std, a=-3*std, b=3*std)
        elif isinstance(m, nn.LayerNorm):
            m.no_weight_decay = True
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)

    def forward(self, atoks, labels=None, noloss=False, flattened=True):
        if flattened:
            atoks = unflatten_snac(atoks)

        assert atoks.shape[-1] == self.ctx_n
        atoks = atoks.long()

        positions = torch.arange(0, atoks.shape[-1], device=atoks.device)

        x = self.embds(atoks, torch.zeros((1,), dtype=torch.float32, device=atoks.device))
        for layer in self.encoder:
            x = layer(x, positions, causal=True)

        emb = self.emb_proj(x[:, -1])
        if noloss: return emb

        class_logits = self.class_head(emb)
        class_loss = F.cross_entropy(
            class_logits,
            labels,
            label_smoothing=self.tunables.label_smoothing if self.training else 0
        )

        logits = self.ar_head(x, embeddings=self.embds.embeddings)
        logits *= self.tunables.output_mult / (self.width / self.base_width)

        ar_loss = 0
        for i in range(self.quantizers):
            ar_loss += F.cross_entropy(
                logits[:,i,:-1].reshape(-1,logits.shape[-1]),
                atoks[:,i,1:].reshape(-1),
                ignore_index=0
            )
        ar_loss /= self.quantizers

        metrics = {
            "class_loss": self.tunables.class_loss_ratio * class_loss,
            "ar_loss": ar_loss
        }

        if not self.training:
            metrics["accuracy"] = (class_logits.argmax(-1) == labels).float().mean()

        return emb, metrics

    def classify(self, emb):
        logits = self.class_head(emb)
        return logits

    def setup(self, device):
        ...

    def save_model(self, fname):
        torch.save(dict(config = self.__stored_args__,
                        tunables = dataclasses.asdict(self.tunables),
                        state_dict = self.state_dict()), fname)

    def load_checkpoint(self, local_filename_or_obj):
        if isinstance(local_filename_or_obj, (str, Path)):
            spec = torch.load(local_filename_or_obj, map_location='cpu')
        else:
            spec = local_filename_or_obj
        assert 'pytorch-lightning_version' in spec, 'not a valid PyTorch Lightning checkpoint'
        state_dict = {k.replace('model.', ''):v
                      for k,v in spec['state_dict'].items()}
        self.load_state_dict(state_dict)
        return self

    @classmethod
    def load_model(cls, ref="streich/artist_emb:artist_emb.model",
                   repo_id=None, filename=None, local_filename=None, spec=None, device=None):
        if repo_id is None and filename is None and local_filename is None and spec is None:
            if ":" in ref:
                repo_id, filename = ref.split(":", 1)
            else:
                local_filename = ref
        if not local_filename and spec is None:
            local_filename = hf_hub_download(repo_id=repo_id, filename=filename)
        if spec is None:
            spec = torch.load(local_filename, map_location=device)

        model = cls(**spec['config'], tunables=Tunables(**spec['tunables']))
        model.load_state_dict(spec['state_dict'])
        model.eval().to(device)
        return model

class EmbeddingDataset(SimpleDataset):
    def __init__(
        self,
        *args,
        inference=False,
        chunk_size=None,
        artist_file=None,
        min_song_count=50,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inference = inference
        self.chunk_size = chunk_size

        if artist_file is not None:
            with open(artist_file) as f:
                self.artists = json.load(f)

    def prepare_df(self, idx):
        atoks = self.load_df(idx, "atoks_vocals_snac_32khz")

        if not self.inference:
            artist = self.load_df(idx, "chunked", columns=["id", "artist"])
            artist = artist[artist["artist"].isin(self.artists)]
            df = pd.concat([atoks, artist], axis=1, join="inner")
        else:
            df = atoks

        df["atoks"] = df["values"].apply(lambda x: torch.tensor(x.reshape(4, -1)))
        chunk_size = self.chunk_size
        if chunk_size is not None:
            full_length = df["atoks"].iloc[0].shape[0]
            expanded = None
            for offset in range(0, full_length, chunk_size):
                sliced = df
                sliced["atoks"] = sliced["atoks"].apply(lambda x: x[:, offset:offset + chunk_size])
                if expanded: expanded = pd.concat([expanded, sliced])
                else: expanded = sliced
            df = expanded
        df["id"] = df.index

        return df

    def entries(self):
        import torch

        for entry in super().entries():
            artist_id = self.artists.get(entry.get("artist", None), None)
            atoks = entry["atoks"]

            flattened = flatten_snac(atoks)
            if self.inference:
                yield {
                    "id": entry["id"],
                    "atoks": chunk,
                }
            else:
                sample = (flattened, torch.tensor(artist_id))
                yield tuple(x.unsqueeze(0).clone() for x in sample)

def load_dataset(
    split,
    path=None,
    num_partitions=8,
    **kwargs
):
    if "artist_file" not in kwargs:
        raise Exception("You need to generate an artist file before training using the `ttv/artist_embeddings.py` script. And add it to the dataset config.")

    subset = [0] if split == "val" else range(1, num_partitions + 1)
    return EmbeddingDataset(
        path,
        subset=subset,
        shufbuf_size=1000,
        **kwargs,
    )

def make_model(dataset, tunables=Tunables(), **kwargs):
    ctx_n = unflatten_snac(next(iter(dataset))[0]).shape[-1]
    return Embedder(num_classes=len(dataset.artists), ctx_n=ctx_n, **kwargs)

def load_model(*args, **kwargs):
    model = Embedder.load_model(*args, **kwargs)
    return model

def collect_artists(dataset, min_count):
    counts = defaultdict(int)
    for idx in dataset.subset:
        for artist in dataset.load_df(idx, "chunked", columns=["artist"])["artist"]:
            counts[artist] += 1

    result = dict()

    idx = 0
    for artist, count in sorted(list(counts.items())):
        if count < min_count: continue
        result[artist] = idx
        idx += 1

    print(f"Found {len(result)} frequent artists in dataset")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_config")
    parser.add_argument("--output-path", default="artist_ids.json")
    parser.add_argument(
        "--min-sample-count",
        type=int, default=100,
        help="Minimum number of samples for an artist to be included"
    )
    args = parser.parse_args()

    dataset_config = json.loads(args.dataset_config)
    dataset = load_dataset("train", artist_file=None, inference=True, **dataset_config)
    artist_ids = collect_artists(dataset, args.min_sample_count)
    with open(args.output_path, "w") as f:
        json.dump(artist_ids, f)



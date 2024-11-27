import math

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from fastcore.basics import store_attr
from fastprogress import progress_bar

from whisperspeech.t2s_up_wds_mlang_enclm import (
    Tunables,
    TSARTransformer,
    CharTokenizer,
)
from whisperspeech import languages
from whisperspeech import inference
from whisperspeech.modules import *

from autosing.alignment import Alignment
from autosing.sm2a import (
    BaseDecoderME,
    DelSumEmbedding,
    apply_pattern,
)
from autosing.codecs import load_codec_model
from autosing.simple_dataset import SimpleDataset

class T2SDataset(SimpleDataset):
    def __init__(
        self,
        dataset_dir,
        *args,
        parts=["main"],
        stoks_pad_token=2048,
        codec_spec="snac_32khz",
        min_inlier_ratio=0.85,
        mask_finegrained=2,
        shufbuf_size=1000,
        **kwargs
    ):
        super().__init__(dataset_dir, *args, **kwargs)

        self.ttoks_len = 450 if "short" in dataset_dir else 900
        self.tokenizer = CharTokenizer()

        self.stoks_pad_token = stoks_pad_token

        self.parts = parts
        self.stoks_pad_token = stoks_pad_token
        self.codec_spec = codec_spec
        self.codec = load_codec_model(codec_spec)
        self.language = languages.to_id("en")
        self.min_inlier_ratio = min_inlier_ratio
        self.mask_finegrained = mask_finegrained

        batch = next(self.entries())
        self.stoks_len = batch[-1].shape[-1]
        self.mtoks_len = batch[-2].shape[-1]
        self.quantizers = batch[-2].shape[-2]

    def prepare_df(self, idx):
        df = pd.concat([self.load_df(idx, part) for part in self.parts])
        return df

    def entries(self):
        for entry in super().entries():
            stoks = torch.tensor(entry["stoks"]) 
            stoks_len = stoks.shape[-1]
            stoks = F.pad(stoks, (1, 0), value=self.stoks_pad_token)

            max_length = 2496

            atoks = torch.tensor(entry["atoks"].reshape(4, -1))
            atoks = atoks[:, :min(max_length, atoks.shape[-1])]

            if "mtoks" in entry and type(entry["mtoks"]) is np.ndarray:
                mtoks = torch.tensor(entry["mtoks"].reshape(4, -1))
                mtoks = mtoks[:, :min(max_length, mtoks.shape[-1])]
            else:
                mtoks = self.codec.num_codes + 0 * atoks
            mtoks = apply_pattern("bricks", mtoks, self.codec.num_codes)

            if self.mask_finegrained:
                mtoks[self.mask_finegrained:] = self.codec.num_codes

            if (
                type(entry["alignment"]) is dict and
                entry["alignment"]["inlier_ratio"] > self.min_inlier_ratio
            ):
                alignment = Alignment(**entry["alignment"]).slice(
                    0.1, stoks_len // 25 - 0.1, blank_partial=True
                )
                words = []
                next_line = False
                for word_start, word in zip(alignment.word_spans[0], alignment.words):
                    if word_start > 15 and not next_line:
                        words.append("\n")
                        next_line = True
                    words.append(word)
                text = " ".join(word for word in words if word != "")
                text = text.replace(" \n ", "\n")
            elif "text" in entry and type(entry["text"]) is str:
                text = entry["text"]
            else:
                text = "@@@@placeholder@@@@@"

            ttoks = torch.tensor(self.tokenizer.encode(text))
            if len(ttoks) > self.ttoks_len: print("not enough ttoks")
            ttoks = ttoks[:min(self.ttoks_len - 1, len(ttoks))]
            ttoks = F.pad(ttoks, (1, self.ttoks_len - ttoks.shape[-1]), value=CharTokenizer.eot)

            cps = 1

            sample = (
                ttoks[:-1], ttoks[1:],
                torch.tensor(self.language), torch.tensor(cps),
                stoks[:-1], mtoks, stoks[1:],
            )
            yield tuple(x.unsqueeze(0) for x in sample)

def load_dataset(split, path=None, num_partitions=512, **kwargs):
    subset = [0] if split == "val" else list(range(1, num_partitions + 1))
    return T2SDataset(
        path, subset,
        shuffle_df=True,
        shufbuf_size=1000,
        **kwargs
    )

# Model definition adapted from WhisperSpeech

class TM2STransformer(TSARTransformer):
    def __init__(
        self,
        *args,
        tunables=Tunables(),
        n_head=3, head_width=64, ffn_mult=4,
        mtoks_len=None, quantizers=4, codes=1024,
        **kwargs,
    ):
        super().__init__(
            *args,
            tunables=Tunables(),
            n_head=n_head,
            head_width=head_width,
            ffn_mult=ffn_mult,
            **kwargs
        )

        store_attr("mtoks_len,quantizers,codes")

        self.music_embds = DelSumEmbedding(
            pos_embs=None, length=mtoks_len,
            n_head=n_head, head_width=head_width, atoks_width=None,
            quantizers=quantizers, codes=self.codes,
        )

        qk_scale = self.tunables.query_mult * 8 / math.sqrt(head_width)
        encoder_depth = int(len(self.encoder.layers))
        self.music_encoder = nn.Sequential(*[
            ResidualAttentionBlock(
                self.width, n_head, qk_scale=qk_scale,
                ffn_mult=ffn_mult, rope=True,
            )
            for _ in range(encoder_depth)
        ])
        self.ln_post_music = LayerNorm(self.width)

        self.decoder = BaseDecoderME(
            qk_scale=qk_scale, length=self.stoks_len,
            n_head=n_head, width=n_head * head_width,
            ffn_mult=ffn_mult, depth=len(self.decoder.layers),
            rope=True
        )

        self.apply(self.init_transformer)

    def _music_encoder(self, memb, positions):
        x = memb
        for layer in self.music_encoder:
            x = layer(x, positions)
        return self.ln_post_music(x)

    def run_music_encoder(self, Mtoks, xenc):
        embs = self.music_embds(Mtoks, xenc)
        positions = torch.arange(0, embs.shape[1], device=embs.device)
        yenc = self._music_encoder(embs, positions)
        enc_logits = None
        return yenc, positions, enc_logits

    def forward(
        self,
        in_ttoks, out_ttoks,
        languages, cpss,
        in_stoks, mtoks, out_stoks=None, in_stoks_positions=None,
        loss=True, offset=None,
        xenc=None, xenc_positions=None,
        yenc=None, yenc_positions=None,
        cps_emb=None
    ):
        in_stoks = in_stoks.to(dtype=torch.long)

        if out_stoks is not None:
            out_stoks = out_stoks.to(dtype=torch.long)

        if xenc is None:
            mtoks, in_ttoks, out_ttoks = tuple(
                x.to(dtype=torch.long)
                for x in [mtoks, in_ttoks, out_ttoks]
            )
            xenc, xenc_positions, cps_emb = self.run_encoder(in_ttoks, languages, cpss)
            yenc, yenc_positions, _ = self.run_music_encoder(mtoks, xenc)

        if in_stoks_positions is None:
            in_stoks_positions = torch.arange(0, in_stoks.shape[1], device=in_stoks.device)

        x = (self.embeddings.embedding(in_stoks) +
             self.embeddings.positional_embedding[in_stoks_positions] +
             cps_emb).to(xenc[0].dtype)

        x = self.decoder(
            x,
            in_stoks_positions,
            xenc.clone(), xenc_positions,
            yenc, yenc_positions,
        )
        logits = self.embeddings.embedding.unembed(x)
        logits = logits * self.tunables.output_mult / (self.width / self.base_width)

        if loss is not None:
            loss = F.cross_entropy(logits.transpose(-1,-2), out_stoks)
            if self.training and self.tunables.causal_encoder:
                enc_logits = self.encoder.embedding.unembed(xenc)
                enc_logits = enc_logits * self.tunables.output_mult / (self.width / self.base_width)
                loss += 0.1 * F.cross_entropy(enc_logits.transpose(-1,-2), out_ttoks)

        return logits, loss

    def optimize(self, max_batch_size=1, dtype=torch.float16, torch_compile=True):
        for emb in [self.embeddings.embedding, self.embeddings.embedding]:
            emb.convert_for_eval()
        for emb in self.music_embds.embeddings:
            emb.convert_for_eval()
        for layer in self.encoder.layers:
            layer.attn.convert_for_eval()
        for layer in self.music_encoder:
            layer.attn.convert_for_eval()
        for layer in self.decoder.layers:
            layer.attn.convert_for_eval()
            layer.cross_attn.convert_for_eval()
            layer.setup_kv_cache(max_batch_size, self.stoks_len, self.ttoks_len, self.mtoks_len)
        self.switch_dtypes(dtype)
        if torch_compile:
            self.generate_next = torch.compile(self.generate_next, mode="reduce-overhead", fullgraph=True)

    def generate_one(
        self,
        toks, toks_positions,
        cps_emb,
        xenc, xenc_positions,
        yenc, yenc_positions,
        T, top_k
    ):
        probs, _ = self(
            None, None, None, None,
            toks, mtoks=None,
            in_stoks_positions=toks_positions, loss=None,
            xenc=xenc, xenc_positions=xenc_positions,
            yenc=yenc, yenc_positions=yenc_positions,
            cps_emb=cps_emb
        )
        probs = probs[:,-1]
        probs[self.embeddings.embedding.codes:] = -torch.inf
        return inference.sample(probs, T, top_k)

    @torch.no_grad()
    def generate(
        self,
        txt, mtoks,
        cps=15, lang="en",
        stoks_prompt=None, N=None,
        bs=1, T=0.7,
        top_k=None, step=None,
        show_progress_bar=True
    ):
        self.ensure_tokenizer()
        N = N or self.stoks_len
        dev = self.device
        ttoks = []
        langs = []
        if isinstance(lang, list):
            lang0 = lang[0]
            assert isinstance(txt, list), "lang and txt have to be both lists or strings"
            for txt, lang in zip(txt, lang):
                tt = self.tokenizer.encode(txt)
                ttoks += tt
                langs += [languages.to_id(lang)] * len(tt)
        elif isinstance(lang, torch.Tensor):
            langs = lang
            ttoks = self.tokenizer.encode(txt)
        else:
            lang0 = lang
            ttoks = self.tokenizer.encode(txt)
            langs = torch.tensor([languages.to_id(lang)], device=dev).unsqueeze(0)
        ttoks = torch.tensor(ttoks, device=dev)
        ttoks = F.pad(ttoks, (1, self.ttoks_len - len(ttoks) - 1), value=self.tokenizer.eot).unsqueeze(0)
        cpss = torch.tensor([cps], device=dev)
        T = torch.tensor(T, device=dev)
        if not isinstance(langs, torch.Tensor):
            langs = torch.tensor(langs, device=dev)
            langs = F.pad(langs, (1, self.ttoks_len - len(langs) - 1), value=languages.to_id(lang0)).unsqueeze(0)

        assert mtoks.shape[1] == self.mtoks_len
        mtoks = mtoks.unsqueeze(0).to(int).to(dev)

        toks = torch.zeros((bs,N), dtype=torch.long, device=dev)
        toks[:,0] = self.stoks_codes-1
        start = 0
        if stoks_prompt is not None:
            toks[:,1:len(stoks_prompt)+1] = stoks_prompt
            start = len(stoks_prompt)
        it = range(start+1,N-1)
        if show_progress_bar: it = progress_bar(it)

        toks_positions = torch.arange(N, device=dev)
        ttoks, langs, cpss = [x.repeat(bs, 1) for x in (ttoks, langs, cpss)]
        xenc, xenc_positions, cps_emb = self.run_encoder(ttoks, langs, cpss)
        yenc, yenc_positions, _ = self.run_music_encoder(mtoks, xenc)
        toks_positions = torch.arange(N+1, device=dev)
        toks[:,start+1] = self.generate_one(
            toks[:,:start+1].contiguous(), toks_positions[:start+1], cps_emb,
            xenc, xenc_positions,
            yenc, yenc_positions,
            T, top_k
        )
        with inference.inference_context():
            for i in it:
                toks[:,i+1] = self.generate_next(
                    toks[:,i:i+1], toks_positions[i:i+1],
                    cps_emb,
                    xenc, xenc_positions,
                    yenc, yenc_positions,
                    T, top_k
                )[:,0]

                # for profiling, debugging or early exit
                if step is not None: step()

        return toks[:,1:]

def _make_model(size:str, tunables:Tunables=Tunables(), dataset=None, **kwargs):
    kwargs = dict(
        stoks_len = dataset.stoks_len,
        ttoks_len = dataset.ttoks_len,
        mtoks_len = dataset.mtoks_len,
        tunables=tunables, **kwargs
    )
    if 'stoks_codes' not in kwargs: kwargs['stoks_codes'] = dataset.stoks_codes
    if size == 'micro':
        return TM2STransformer(depth=2, n_head=3, ffn_mult=1, **kwargs)
    if size == 'tiny':
        return TM2STransformer(depth=4, n_head=6, **kwargs)
    if size == 'base':
        return TM2STransformer(depth=6, n_head=8, **kwargs)
    if size == 'small':
        return TM2STransformer(depth=12, n_head=12, **kwargs)
    if size == 'small+':
        return TM2STransformer(depth=12, n_head=16, **kwargs)
    if size == 'medium':
        return TM2STransformer(depth=24, n_head=16, **kwargs)

def load_base_params(model, base_path):
    base_model = TSARTransformer.load_model(base_path)
    del base_model.embeddings
    del base_model.encoder.positional_embedding

    pretrained_params = base_model.state_dict()
    state_dict = {**model.state_dict(), **pretrained_params}
    model.load_state_dict(state_dict)
    return model

def make_model(
    size:str,
    frozen_embeddings_model:str="streich/bn_whisper",
    tunables:Tunables=Tunables(),
    dataset=None,
    base_model=None,
):
    import autosing.bn_whisper as bn_whisper

    codec = load_codec_model(dataset.codec_spec)

    if frozen_embeddings_model:
        vqmodel = bn_whisper.load_model(frozen_embeddings_model)
        model = _make_model(
            size,
            tunables, dataset,
            stoks_codes=vqmodel.vq_codes+1,
            stoks_width=vqmodel.rq.layers[0]._codebook.embed[0].shape[-1],
            quantizers=dataset.quantizers, codes=codec.num_codes,
        )
        model.load_frozen_semantic_embeddings(vqmodel)
    else:
        model = _make_model(size, tunables, dataset)

    if base_model is not None:
        load_base_params(model, base_model)

    return model

def load_model(*args, **kwargs):
    model = TM2STransformer.load_model(*args, **kwargs)
    return model


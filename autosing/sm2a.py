import math
import dataclasses
from typing import Optional
from random import Random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import record_function

from fastprogress import progress_bar
from fastcore.basics import store_attr

from whisperspeech.s2a_delar_mup_wds_mlang import (
    SADelARTransformer,
    DelSumEmbedding,
    DelSumHead,
    Tunables as S2ATunables
)
from whisperspeech.modules import (
    MultiHeadAttention,
    ResidualAttentionBlock,
    LayerNorm,
)
from whisperspeech import inference

import autosing
from autosing.codecs import load_codec_model
from autosing.simple_dataset import SimpleDataset

class SM2ADataset(SimpleDataset):
    def __init__(
        self,
        *args,
        parts=["main"],
        stoks_pad_token=2048,
        codec_spec="snac_32khz",
        p_no_cond=0.0,
        multiscale=False,
        p_coarse=0.75,
        shufbuf_size=1000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.parts = parts
        self.stoks_pad_token = stoks_pad_token
        self.codec_spec = codec_spec
        self.codec = load_codec_model(codec_spec)
        self.p_no_cond = p_no_cond
        self.multiscale = multiscale
        self.p_coarse = p_coarse

        batch = next(self.entries())
        (
            self.stoks_len,
            self.atoks_len,
            self.mtoks_len,
            self.spk_width,
            _,
        ) = tuple(x.shape[-1] for x in batch)
        self.quantizers = batch[1].shape[-2]

    def prepare_df(self, idx):
        return pd.concat([self.load_df(idx, part) for part in self.parts])

    def entries(self):
        rand = Random(42 + self.shard_info()[0])

        for entry in super().entries():
            stoks = torch.tensor(entry["stoks"]) 
            stoks = stoks[:-1]
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

            pattern = (
                "bricks"
                if self.multiscale and rand.uniform(0, 1) < self.p_coarse else
                "bricks_unrolled"
            )

            atoks = apply_pattern(pattern, atoks, self.codec.num_codes)

            if self.multiscale and pattern == "bricks_unrolled":
                atoks_len = mtoks.shape[-1]
                offset = rand.randint(0, atoks.shape[-1] - atoks_len)
                offset = offset // 8 * 8
                atoks = atoks[:, offset:offset + atoks_len]
                atoks_positions = offset + torch.arange(atoks.shape[-1])
            elif self.multiscale and pattern == "bricks":
                atoks_positions = 4 * torch.arange(atoks.shape[-1])
            else:
                atoks_positions = torch.arange(atoks.shape[-1])

            assert atoks_positions.shape[-1] == atoks.shape[-1]

            if rand.uniform(0, 1) < self.p_no_cond:
                mtoks = self.codec.num_codes + 0 * mtoks

            artist_embs = torch.tensor(entry["embs"].reshape(-1, 256))
            if artist_embs.shape[0] == 1:
                artist_embs = artist_embs.squeeze()
            else:
                artist_embs = artist_embs.mean(0)

            if rand.uniform(0, 1) < self.p_no_cond:
                artist_embs *= 0

            sample = (
                stoks,
                atoks,
                mtoks,
                artist_embs,
                atoks_positions,
            )

            yield tuple(x.unsqueeze(0) for x in sample)

def load_dataset(split, path=None, num_partitions=512, partition_offset=0, **kwargs):
    subset = [0] if split == "val" else list(range(partition_offset + 1, num_partitions + partition_offset + 1))
    return SM2ADataset(
        path, subset,
        shuffle_df=True,
        **kwargs
    )

@dataclasses.dataclass
class Tunables(S2ATunables):
    music_encoder_ratio: float = 1
    causal_music_encoder: bool = False

# Components from whisperspeech.modules adapted for multiple cross attention signals

class ResidualAttentionBlockME(ResidualAttentionBlock):
    def __init__(
        self,
        n_state, n_head,
        qk_scale=1, rope=False,
        ffn_mult=4, alpha=0.8,
    ):
        super().__init__(
            n_state, n_head,
            cross_attention=True,
            qk_scale=qk_scale, rope=rope,
            ffn_mult=ffn_mult,
        )
        self.cross_attn_two = MultiHeadAttention(
            n_state, n_head, qk_scale=qk_scale, rope=rope, cross=True
        )
        self.alpha = alpha
    
    def setup_kv_cache(
        self,
        max_batch_size,
        max_seq_len,
        max_cross_seq_len=None,
        max_cross_seq_len_two=None
    ):
        self.attn.setup_kv_cache(max_batch_size, max_seq_len)
        self.cross_attn.setup_kv_cache(max_batch_size, max_cross_seq_len)
        self.cross_attn_two.setup_kv_cache(max_batch_size, max_cross_seq_len_two)
    
    def forward(
        self,
        x: Tensor,
        x_positions: Tensor=None,
        xa: Optional[Tensor]=None,
        xa_positions: Optional[Tensor]=None,
        xb: Optional[Tensor]=None,
        xb_positions: Optional[Tensor]=None,
        causal=False,
        mask=None,
    ):
        lnx = self.attn_ln(x)
        x = x + self.attn(lnx, x_positions, lnx, x_positions, causal=causal, mask=mask)
        lnx = self.cross_attn_ln(x)
        u = self.cross_attn(lnx, x_positions, xa, xa_positions)
        v = self.cross_attn_two(lnx, x_positions, xb, xb_positions)
        x = x + self.alpha * u + (1 - self.alpha) * v
        x = x + self.mlp(self.mlp_ln(x))
        return x

class BaseDecoderME(nn.Module):
    def __init__(
        self,
        depth=6, n_head=6, width=384,
        qk_scale=1, ffn_mult=4, length=2250, rope=False,
    ):
        super().__init__()

        self.length = length
        self.width = width
        self.layers = nn.ModuleList([
            ResidualAttentionBlockME(
                self.width, n_head,
                qk_scale=qk_scale, ffn_mult=ffn_mult, rope=rope,
            ) for _ in range(math.floor(depth))
        ])
        self.ln_post = LayerNorm(width)
        mask = torch.empty(length, length).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, x_positions, xenc, xenc_positions, yenc, yenc_positions, **kwargs):
        for i, layer in enumerate(self.layers):
            x = layer(
                x, x_positions,
                xenc, xenc_positions,
                yenc, yenc_positions,
                causal=self.training,
                mask=self.mask if not self.training else None,
                **kwargs,
            )
        x = self.ln_post(x)
        return x

# Model definition adapted from WhisperSpeech

class SM2ATransformer(SADelARTransformer):
    def __init__(
        self,
        *args,
        tunables=Tunables(), ctx_n=2250,
        n_head=3, head_width=64, ffn_mult=4, atoks_width=None,
        mtoks_len=None, pattern=None,
        enc_subsample=1,
        **kwargs,
    ):
        super().__init__(
            *args,
            tunables=tunables,
            ctx_n=ctx_n,
            n_head=n_head,
            head_width=head_width,
            ffn_mult=ffn_mult,
            atoks_width=atoks_width,
            **kwargs
        )

        self.tunables = tunables
        assert not self.tunables.causal_encoder

        mtoks_len = mtoks_len or self.ctx_n
        self.mtoks_len = mtoks_len
        self.pattern = pattern
        store_attr("mtoks_len,pattern,enc_subsample")

        qk_scale = self.tunables.query_mult * 8 / math.sqrt(head_width)

        encoder_depth = int(len(self.encoder) * tunables.music_encoder_ratio)
        self.music_encoder = nn.Sequential(*[
            ResidualAttentionBlock(
                self.width, n_head, qk_scale=qk_scale,
                ffn_mult=ffn_mult, rope=self.tunables.rope
            )
            for _ in range(encoder_depth)
        ])
        self.ln_post_music = LayerNorm(self.width)

        if self.tunables.causal_music_encoder:
            self.music_embds = DelSumEmbedding(
                pos_embs=self.positional_embeddings, length=ctx_n,
                n_head=n_head, head_width=head_width, atoks_width=atoks_width,
                quantizers=self.quantizers, codes=self.codes,
            )
            self.music_head = DelSumHead(
                n_head=n_head,
                head_width=head_width,
                quantizers=self.quantizers
            )

        # Overwrite the existing decoder with our multi-encoder one
        self.decoder = BaseDecoderME(
            qk_scale=qk_scale, length=ctx_n,
            n_head=n_head, width=n_head * head_width, 
            ffn_mult=ffn_mult, depth=len(self.decoder.layers),
            rope=tunables.rope,
        )

        for layer in self.decoder.layers:
            layer.cross_attn.key_subsampling = enc_subsample

        self.apply(self.init_transformer)

    def _music_encoder(self, memb, positions):
        x = memb
        for layer in self.music_encoder:
            x = layer(x, positions, causal=self.tunables.causal_music_encoder)
        return self.ln_post_music(x)

    def run_music_encoder(self, Mtoks, xenc):
        if self.tunables.causal_music_encoder:
            embs = self.music_embds(Mtoks, xenc)
        else:
            embs = self.embds(Mtoks, xenc)

        with record_function("encoder"):
            if self.positional_embeddings is not None:
                raise NotImplementedError()
            positions = torch.arange(0, embs.shape[1], device=embs.device)
            yenc = self._music_encoder(embs, positions)

        if self.training and self.tunables.causal_music_encoder:
            enc_logits = self.music_head(yenc, embeddings=self.music_embds.embeddings)
            enc_logits *= self.tunables.output_mult / (self.width / self.base_width)
        else:
            enc_logits = None

        return yenc, positions, enc_logits

    def logits_from_hidden(self, x):
        logits = self.head(x, embeddings=self.embds.embeddings)
        logits *= self.tunables.output_mult / (self.width / self.base_width)
        return logits

    def forward(
        self,
        Stoks, Atoks, Mtoks, spk_embs, atoks_positions=None,
        langs=None, out_stoks=None, out_atoks=None,
        noloss=False,
        xenc=None, xenc_positions=None,
        yenc=None, yenc_positions=None,
        ret_hidden=False,
    ):
        if xenc is None:
            Stoks, Atoks, Mtoks = [x.to(dtype=torch.long) for x in (Stoks, Atoks, Mtoks)]
            xenc, xenc_positions, enc_logits = self.run_encoder(Stoks, spk_embs)
            yenc, yenc_positions, music_logits = self.run_music_encoder(Mtoks, xenc)
        with record_function("decoder"):
            embs = self.embds(Atoks, xenc)
            if atoks_positions is None:
                atoks_positions = torch.arange(0, embs.shape[1], device=embs.device)
            x = self.decoder(
                embs,
                atoks_positions,
                xenc, xenc_positions,
                yenc, yenc_positions,
            )

        logits = self.logits_from_hidden(x)

        if ret_hidden:
            return logits, x

        if noloss:
            return logits

        with record_function("loss"):
            loss = 0
            for i in range(self.quantizers):
                loss += F.cross_entropy(
                    logits[:,i,:-1].reshape(-1,logits.shape[-1]),
                    Atoks[:,i,1:].reshape(-1),
                    ignore_index=self.codes
                )
                if self.training and i == 0:
                    loss *= self.tunables.q0_loss_mult
            loss_denom = self.quantizers
            if self.training: loss_denom += - 1 + self.tunables.q0_loss_mult
            loss /= loss_denom

            if self.training and self.tunables.causal_encoder:
                loss += 0.1 * F.cross_entropy(enc_logits[:,:-1].transpose(-1,-2), Stoks[:,1:])

            if self.training and self.tunables.causal_music_encoder:
                for i in range(self.quantizers):
                    loss += F.cross_entropy(
                        music_logits[:,i,:-1].reshape(-1,music_logits.shape[-1]),
                        Mtoks[:,i,1:].reshape(-1),
                        ignore_index=self.codes
                    ) / self.quantizers

        if not self.training:
            for i in range(self.quantizers):
                Atoks_i = Atoks[:,i,1:]
                valid_Atoks = Atoks_i != self.codes
                self.val_true[i] += (logits[:,i,:-1].argmax(-1)[valid_Atoks] == Atoks_i[valid_Atoks]).float().sum()
                self.val_total[i] += valid_Atoks.float().sum()

        return logits, loss

    def optimize(self, max_batch_size=1, dtype=torch.float16, torch_compile=True):
        for emb in self.embds.embeddings:
            emb.convert_for_eval()

        for layer in self.encoder:
            layer.attn.convert_for_eval()
        for layer in self.music_encoder:
            layer.attn.convert_for_eval()
        for layer in self.decoder.layers:
            layer.attn.convert_for_eval()
            layer.cross_attn.convert_for_eval()
            layer.setup_kv_cache(max_batch_size, self.ctx_n, self.stoks_len, self.mtoks_len)
        self.switch_dtypes(dtype)
        if torch_compile:
            self.generate_next = torch.compile(self.generate_next, mode="reduce-overhead", fullgraph=True)

    def optimize_training(self):
        self.decoder = torch.compile(self.decoder, fullgraph=True, mode="reduce-overhead")
        self._encoder = torch.compile(self._encoder, fullgraph=True, mode="reduce-overhead")

    def generate_one(
        self,
        toks, positions, langs,
        xenc, xenc_positions,
        yenc, yenc_positions,
        T, top_k, cfg_lambda=None,
    ):
        probs = self(
            None, toks, None, None, noloss=True,
            xenc=xenc, xenc_positions=xenc_positions,
            yenc=yenc, yenc_positions=yenc_positions,
            atoks_positions=positions,
        )[:,:,-1]

        if cfg_lambda is not None:
            probs = (1 + cfg_lambda) * probs[1:] - cfg_lambda * probs[:1]

        result = torch.stack([
            inference.sample(probs[:, i], (1 if i < 20 else 1.25) * T, top_k)
            for i in range(self.quantizers)
        ], 1)
        return result

    @torch.no_grad()
    def generate(
        self,
        stoks, mtoks, speakers,
        langs=None,
        atoks_prompt=None,
        N=None, bs=1, T=0.7, top_k=None,
        show_progress_bar=True, step=None,
        subsample_enc=False,
        template=None,
        cfg_lambda=None,
    ):
        dev = self.device
        N = N or template.shape[-1]
        stoks = F.pad(
            stoks.to(dev),
            (1, self.stoks_len - len(stoks) - 1),
            value=self.stoks_codes-1
        ).unsqueeze(0)

        assert mtoks.shape[1] == self.mtoks_len
        mtoks = mtoks.clone().unsqueeze(0).to(int).to(dev)
        speakers = speakers.unsqueeze(0).to(self.dtype).to(dev)

        if template is None:
            template = mtoks

        toks = torch.full(
            (bs,self.quantizers,self.ctx_n),
            self.codes+1,
            dtype=torch.long, device=dev
        )
        T = torch.tensor(T, device=dev)

        template = template.to(toks.device).to(toks.dtype)

        if cfg_lambda is not None:
            cfg_lambda = torch.tensor(cfg_lambda, device=dev)
            mtoks = torch.cat([torch.zeros_like(mtoks), mtoks], 0)
            stoks = torch.cat([stoks, stoks], 0)
            speakers = torch.cat([speakers, speakers], 0)

        start = 0 # number of valid tokens or the index of first empty spot
        if atoks_prompt is not None:
            start = atoks_prompt.shape[-1]
            for i in range(self.quantizers):
                toks[:,i,1+i:start+i+1] = atoks_prompt[:,i]
        start += 1 # we always start with at least an SOT

        toks[:, template[0] >= self.codes] = template[template >= self.codes]

        with record_function("encode"):
            stoks, speakers = [x.repeat(bs, 1) for x in (stoks, speakers)]
            xenc, xenc_positions, _ = self.run_encoder(stoks, speakers)
            yenc, yenc_positions, _ = self.run_music_encoder(mtoks, xenc)
            toks_positions = torch.arange(N, device=dev)
        with record_function("prefill"):
            initial = self.generate_one(
                toks[:,:,:start], toks_positions[:start], langs,
                xenc, xenc_positions,
                yenc, yenc_positions,
                T, top_k, cfg_lambda=cfg_lambda,
            )
            toks[:,:,start:start+1] = initial
            start += 1
            toks[:, template[0] >= self.codes] = template[template >= self.codes]

        with inference.inference_context():
            it = range(start,min(N,self.ctx_n-1))
            if show_progress_bar: it = progress_bar(it)

            for i in it:
                with record_function("generate_one"):
                    toks[:,:,i:i+1] = self.generate_next(
                        toks[:,:,i-1:i], toks_positions[i-1:i], langs,
                        xenc, xenc_positions,
                        yenc, yenc_positions,
                        T, top_k, cfg_lambda=cfg_lambda,
                    )

                toks[:, template[0] >= self.codes] = template[template >= self.codes]

                # for profiling, debugging or early exit
                if step is not None: step()

        return remove_pattern(self.pattern, toks[:, :, :N])

    @classmethod
    def load_model(cls, ref="collabora/whisperspeech:s2a-q4-small-en+pl.model",
                   repo_id=None, filename=None, local_filename=None, spec=None, device=None):
        if repo_id is None and filename is None and local_filename is None and spec is None:
            if ":" in ref:
                repo_id, filename = ref.split(":", 1)
            else:
                local_filename = ref
        if not local_filename and spec is None:
            raise NotImplementedError()
        if spec is None:
            spec = torch.load(local_filename, map_location=device)
        if '_extra_state' not in spec['state_dict'] and 'speaker_map' in spec['config']: spec['state_dict']['_extra_state'] = { 'speaker_map': spec['config']['speaker_map'] }

        model = cls(**spec['config'], tunables=Tunables(**Tunables.upgrade(spec['tunables'])))
        model.load_state_dict(spec['state_dict'])
        model.eval().to(device)
        return model

def apply_pattern(pattern, toks, num_codes):
    toks = toks.clone()

    assert len(toks.shape) == 2

    if pattern == "bricks":
        blocks = toks.reshape(4, -1, 4)
        flattened = torch.cat([
            blocks[0, :, :1],
            blocks[1, :, :1],
            blocks[2, :, ::2],
            blocks[3],
        ], -1).transpose(0, 1)
        flattened[0, 1::2] = num_codes
        padding = torch.full((flattened.shape[0], 8), num_codes + 1, dtype=flattened.dtype)
        n = padding.shape[-1] + flattened.shape[-1]
        flattened = torch.cat([padding, flattened, padding], -1)
        result = torch.full_like(flattened[:, :n], -1)
        result[0] = flattened[0, 7:][:n]
        result[1] = flattened[1, 6:][:n]
        result[2:4] = flattened[2:4, 5:][:, :n]
        result[4:] = flattened[4:, 4:][:, :n]
        assert not (result == -1).any()
        return result
    elif pattern == "bricks_unrolled":
        result = torch.repeat_interleave(apply_pattern("bricks", toks, 4096), 4, dim=-1)
        n = torch.arange(result.shape[-1])
        result[:3, n % 4 != 0] = num_codes
        result[3, n % 4 != 2] = num_codes
        for i in range(4):
            result[4 + i, n % 4 != i] = num_codes
        return result
    else:
        raise Exception(f"Invalid pattern {pattern}")

def remove_pattern(pattern, toks):
    assert len(toks.shape) == 3

    bs = toks.shape[0]

    if pattern == "bricks":
        result = torch.zeros((toks.shape[0], 4, 4 * (toks.shape[-1] - 8)), dtype=int)
        result[:, 0, ::8] = toks[:, 0, 1:-7:2]
        result[:, 1, ::4] = toks[:, 1, 2:-6]
        result[:, 2, ::2] = toks[:, 2:4, 3:-5].transpose(1, 2).reshape(bs, -1)
        result[:, 3] = toks[:, 4:, 4:-4].transpose(1, 2).reshape(bs, -1)
        return result
    elif pattern == "bricks_unrolled":
        result = torch.zeros((toks.shape[0], 8, toks.shape[-1] // 4), dtype=toks.dtype)
        result[:, :3] = toks[:, :3, ::4]
        result[:, 3] = toks[:, 3, 2::4]
        for i in range(4):
            result[:, 4 + i] = toks[:, 4 + i, i::4]
        result = remove_pattern("bricks", result)
        return result
    else:
        raise Exception(f"Invalid pattern {pattern}")
            
def _make_model(size:str, quantizers:int=4, tunables=Tunables(), **kwargs):
    kwargs = dict(quantizers=quantizers, tunables=tunables, **kwargs)
    if size == 'tiny':
        return SM2ATransformer(depth=4, n_head=6, **kwargs)
    if size == 'base':
        return SM2ATransformer(depth=6, n_head=8, **kwargs)
    if size == 'small':
        return SM2ATransformer(depth=12, n_head=12, **kwargs)
    if size == 'baseline':
        return SM2ATransformer(depth=20, n_head=16, **kwargs)
    if size == 'medium':
        return SM2ATransformer(depth=24, n_head=16, **kwargs)
    if size == 'large':
        return SM2ATransformer(depth=33, n_head=20, **kwargs)

def load_base_params(model, base_path, num_codes):
    base_model = autosing.s2a.load_model(base_path).cuda()
    del base_model.hidden_to_emb
    del base_model.semantic_embedding
    del base_model.spk_to_hidden
    pretrained_params = base_model.state_dict()

    if pretrained_params["embds.embeddings.0.main.weight"].shape[0] != num_codes:
        for i in range(4):
            del pretrained_params[f"embds.embeddings.{i}.main.weight"]
        del pretrained_params["head.splitter.0.bias"]
        del pretrained_params["head.splitter.0.weight"]
        del pretrained_params["val_true"]
        del pretrained_params["val_total"]

    state_dict = {**model.state_dict(), **pretrained_params}
    model.load_state_dict(state_dict)
    return model

def make_model(
    size:str, tunables:Tunables=Tunables(),
    frozen_embeddings_model="streich/bn_whisper",
    frozen_acoustic_embeddings=False,
    dataset=None,
    base_model=None,
    **kwargs,
):
    import autosing.bn_whisper as bn_whisper

    codec = load_codec_model(dataset.codec_spec)

    vqmodel = (
        bn_whisper.load_model(frozen_embeddings_model)
        if frozen_embeddings_model else
        None
    )

    stoks_codes = vqmodel.vq_codes + 1
    model = _make_model(
        size, dataset.quantizers, tunables,
        stoks_codes=stoks_codes,
        stoks_width=vqmodel.rq.layers[0]._codebook.embed[0].shape[-1],
        codes=codec.num_codes,
        ctx_n=dataset.atoks_len,
        stoks_len=dataset.stoks_len,
        mtoks_len=dataset.mtoks_len,
        spk_width=dataset.spk_width,
        **kwargs,
    )
    model.load_frozen_semantic_embeddings(vqmodel)

    if frozen_acoustic_embeddings:
        from encodec.model import EncodecModel
        amodel = EncodecModel.encodec_model_24khz()
        model.load_frozen_acoustic_embeddings(amodel)

    if base_model is not None:
        model = load_base_params(model, base_model, codec.num_codes)

    return model

def load_model(*args, **kwargs):
    model = SM2ATransformer.load_model(*args, **kwargs)
    model.semantic_embedding.lr_scale = 0
    return model


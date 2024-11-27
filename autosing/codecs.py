import dataclasses
import math
from typing import Any

import torch

@dataclasses.dataclass
class Codec:
    encode: Any
    decode: Any
    sample_rate: int
    frame_rate: int
    num_channels: int
    quantizers: int
    num_codes: int
    alignment: int

def load_codec_model(spec):
    if "snac" in spec:
        from snac import SNAC
        model = SNAC.from_pretrained(f"hubertsiuzdak/{spec}").eval()

        num_channels = 1
        quantizers = 4

        def encode(audio):
            nonlocal model
            model = model.cuda()

            lcm = math.lcm(model.vq_strides[0], model.attn_window_size or 1)
            pad_to = model.hop_length * lcm
            num_frames = (audio.shape[-1] // pad_to) * pad_to
            audio = audio[:, :1, :num_frames].cuda()

            with torch.inference_mode():
                sparse_atoks = model.encode(audio)
                k = len(sparse_atoks)
                atoks = torch.stack([
                    torch.zeros_like(sparse_atoks[-1]) for _ in range(k)
                ], 1)
                for i in range(k):
                    s = 2 ** (k - i - 1)
                    atoks[:, i, ::s] = sparse_atoks[i]

                return atoks.cpu()

        def decode(toks):
            nonlocal model
            model = model.cuda()

            codes = []
            for i in range(4):
                x = toks[i, ::2 ** (3 - i)]
                k = 12 * 2 ** i
                x = x[:(x.shape[-1] // k) * k]
                codes.append(x[None].to(int).cuda())

            with torch.inference_mode():
                return model.decode(codes).cpu()[0]

        return Codec(
            encode, decode,
            model.sampling_rate, model.sampling_rate / model.hop_length,
            num_channels, quantizers, 4096, 8,
        )
    elif spec == "musicgen_32khz":
        from audiocraft.models import CompressionModel
        model = CompressionModel.get_pretrained('facebook/encodec_32khz')

        def encode(audio):
            nonlocal model
            model = model.cuda()
            with torch.inference_mode():
                return model.encode(audio)[0].cpu()

        def decode(atoks):
            nonlocal model
            model = model.cuda()
            with torch.inference_mode():
                return model.decode(atoks.cuda().to(int))[0].cpu()

        return Codec(
            encode, decode,
            model.sample_rate, model.frame_rate,
            model.channels, 4, 2048, 1
        )
    else:
        raise Exception(f"Invalid codec spec {spec}")

def flatten_snac(toks):
    q = 15
    flattened = torch.zeros(
        (q, toks.shape[-1] // 8),
        device=toks.device, dtype=toks.dtype
    )
    count = 0
    for i in range(4):
        k = 2 ** i
        for j in range(k):
            p = (8 // k) * j
            flattened[count] = toks[i, p::8][:flattened.shape[-1]]
            count += 1
    assert count == q
    return flattened

def unflatten_snac(flattened):
    result = torch.zeros(
        (flattened.shape[0], 4, flattened.shape[-1] * 8),
        dtype=flattened.dtype, device=flattened.device
    )
    count = 0
    for i in range(4):
        k = 2 ** i
        for j in range(k):
            p = (8 // k) * j
            result[:, i, p::8] = flattened[:, count]
            count += 1
    return result


import os
import argparse
from pathlib import Path
import gc


import torch
import torchaudio
from einops import rearrange

from audio_separator.separator import Separator

import autosing.t2s
import autosing.sm2a
import autosing.bn_whisper
import autosing.artist_embeddings
from autosing.codecs import load_codec_model
from autosing.sm2a import apply_pattern

codec = load_codec_model("snac_32khz")

def compute_spk_emb(spkfile):
    spk_audio, sample_rate = torchaudio.load(spkfile)
    spk_audio = spk_audio[:, int(sample_rate * args.start_time):]
    spk_audio = spk_audio[:, :min(spk_audio.shape[-1], int(32 * sample_rate))]
    spk_audio = torchaudio.functional.resample(spk_audio, sample_rate, codec.sample_rate)
    atoks = codec.encode(spk_audio[None, :, :30 * codec.sample_rate])
    atoks = rearrange(atoks, "b q (n t) -> (b n) q t", n=3)
    emb_model = autosing.artist_embeddings.load_model("../artist_embeddings_v2.model").cuda()
    with torch.inference_mode():
        batched_embs = emb_model(atoks.cuda(), noloss=True, flattened=False).cpu()
        spk_emb = rearrange(batched_embs, "(b n) ... -> b n ...", n=3).mean(1)[0]
    del emb_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Computed Embedding")

    return spk_emb

def sing(args):

    src_sep_dir = Path("tmp")
    ref = Path(args.reference)

    try:
        mfile = next(src_sep_dir.glob(ref.stem + "_(Instrumental)_*.wav"))
        semfile = next(src_sep_dir.glob(ref.stem + "_(Vocals)_*.wav"))
    except StopIteration:
        separator = Separator(
            output_dir=src_sep_dir,
            model_file_dir="/p/home/jusers/streich1/juwels/myhome/audio-separator-models",
        )
        separator.load_model()
        mfile, semfile = tuple(src_sep_dir / x for x in separator.separate(args.reference))
        del separator
        gc.collect()
        torch.cuda.empty_cache()

    max_atok_len = 2496

    instrumental, sample_rate = torchaudio.load(mfile)
    instrumental = instrumental[:, int(sample_rate * args.start_time):]
    instrumental = instrumental[:, :min(instrumental.shape[-1], int(max(args.length, 32) * sample_rate))]
    instrumental = torchaudio.functional.resample(instrumental, sample_rate, codec.sample_rate)
    mtoks = codec.encode(instrumental[None])[0, :, :max_atok_len]
    template = apply_pattern("bricks_unrolled", mtoks, codec.num_codes)[None]

    if args.tts: mtoks[:] = codec.num_codes

    spk_emb = compute_spk_emb(semfile)
    if args.spkfile is not None:
        spk_emb = (1 - args.spk_alpha) * spk_emb + args.spk_alpha * compute_spk_emb(args.spkfile)

    if not args.gen_semantic or args.semfile:
        semfile = args.semfile or semfile

        sem_model = autosing.bn_whisper.load_model("../bn_whisper_v1.model").cuda()
        sem_model.whisper_model_name = "../ft_whisper_small_improved.pt"
        sem_model.ensure_whisper()

        audio, sample_rate = torchaudio.load(semfile)
        audio = audio[:, int(sample_rate * args.start_time):]
        audio = audio[:, :min(audio.shape[-1], int(32 * sample_rate))]
        audio = torchaudio.functional.resample(
            audio, sample_rate, 16000
        )[:, :30 * 16000].reshape(2, -1)

        with torch.inference_mode():
            stoks = sem_model.encode_audio(audio).cpu().reshape(-1)

        del sem_model
        gc.collect()
        torch.cuda.empty_cache()

        print("Extracted Stoks")
    else:
        tm2s_mtoks = apply_pattern("bricks", mtoks, codec.num_codes)
        # Mask finegrained tokens
        tm2s_mtoks[2:] = codec.num_codes

        tm2s_model = autosing.t2s.TM2STransformer.load_model(
            args.tm2s_model
        ).cuda()
        tm2s_model.mtoks_len = tm2s_mtoks.shape[-1]
        tm2s_model.optimize(torch_compile=False)

        stoks = None

        with torch.inference_mode():
            stoks = tm2s_model.generate(
                args.lyrics,
                tm2s_mtoks, 
                T=args.temp,
                cps=1,
                N=750 + 1,
            )[0]

        del tm2s_model
        gc.collect()
        torch.cuda.empty_cache()

        print("Generated Stoks")

    sm2a_model = autosing.sm2a.load_model(args.sm2a_model).cuda()
    sm2a_model.pattern = "bricks_unrolled"
    sm2a_model.ctx_n = template.shape[-1]
    sm2a_model.optimize(torch_compile=False, max_batch_size=args.batch_size)

    # Have to round this up to 8 because of the snac encoding
    N = int((args.length * codec.frame_rate) // codec.alignment * codec.alignment)

    sm2a_mtoks = apply_pattern("bricks", mtoks, codec.num_codes)

    with torch.inference_mode():
        atoks = sm2a_model.generate(
            stoks,
            sm2a_mtoks,
            spk_emb,
            N=N,
            T=args.temp,
            bs=args.batch_size,
            template=template,
        )

    atoks = atoks.clone()
    atoks[atoks >= codec.num_codes] = 0

    print("Generated Atoks")

    del sm2a_model
    gc.collect()
    torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample_atoks in enumerate(atoks):
        vocals = codec.decode(sample_atoks)

        waveform = vocals + args.music_volume * instrumental[:, :vocals.shape[-1]]

        torchaudio.save(
            output_dir / f"{Path(args.reference).stem}_{args.lyrics_idx}_{i}.mp3",
            waveform,
            codec.sample_rate,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--sm2a-model", default="models/sm2a.model")
    parser.add_argument("--tm2s-model", default="models/t2s.model")
    parser.add_argument("--music-volume", type=float, default=0.8)
    parser.add_argument("--lyrics", default=None)
    parser.add_argument("--start-time", type=int, default=0)
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--spkfile", default=None)
    parser.add_argument("--gen-semantic", type=bool, default=True)
    parser.add_argument("--semfile", type=str, default=None)
    parser.add_argument("--tts", type=bool, default=False)
    parser.add_argument("--output-dir", default="sing_output")
    parser.add_argument("--spk-alpha", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    if args.tts:
        args.music_volume = 0

    print(args)

    if os.path.isdir(args.reference):
        for path in Path(args.reference).glob("*.mp3"):
            args.reference = path
            sing(args)
    else:
        sing(args)


import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

@dataclass
class Alignment:
    starts: np.ndarray
    ends: np.ndarray
    scores: np.ndarray
    tokens: np.ndarray
    words: List[str]
    lines: List[str]
    tokens_per_word: np.ndarray
    words_per_line: np.ndarray

    inlier_ratio: Optional[float]
    lrclib_offset: Optional[float]

    def __post_init__(self):
        if not hasattr(self, "word_spans"):
            self.word_spans = self.sub_spans(self.tokens_per_word)

        if not hasattr(self, "line_spans"):
            tokens_per_line = []

            idx = 0 
            for num in self.words_per_line:
                tokens_per_line.append(sum(self.tokens_per_word[idx:idx+num]))
                idx += num

            self.line_spans = self.sub_spans(np.array(tokens_per_line))

    def sub_spans(self, tokens_per):
        indices = np.insert(np.cumsum(tokens_per), 0, 0).astype(int)
        starts = np.append(self.starts, [0])[indices[:-1]]
        ends = np.insert(self.ends, 0, 0)[indices[1:]]

        for i, num in enumerate(tokens_per):
            if num == 0:
                starts[i] = float("nan")
                ends[i] = float("nan")

        return starts, ends - starts

    def slice(self, start, end, blank_partial=False):
        token_mask = (start < self.starts) & (self.ends < end)

        def slice_grouping(items, counts, mask):
            new_items = []
            new_counts = []
            indices = []

            idx = 0
            for i, (item, count) in enumerate(zip(items, counts)):
                overlap = mask[idx:idx+count].sum()
                idx += count

                if overlap == 0: continue

                blank = overlap != count if blank_partial else False
                new_items.append(item if not blank else "")
                new_counts.append(overlap)
                indices.append(i)

            return new_items, new_counts, indices

        words, tokens_per_word, word_indices = slice_grouping(
            self.words, self.tokens_per_word, token_mask
        )

        word_mask = np.full(len(self.words), False)
        word_mask[word_indices] = True

        lines, words_per_line, _ = slice_grouping(self.lines, self.words_per_line, word_mask)

        return Alignment(
            self.starts[token_mask] - start,
            self.ends[token_mask] - start,
            self.scores[token_mask],
            self.tokens[token_mask],
            words,
            lines,
            tokens_per_word,
            words_per_line,
            self.inlier_ratio,
            self.lrclib_offset,
        )

    def cover(self):
        return self.word_spans[1].sum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = dict()

def compute_emission(audio_file, model_name=None, ret_frame_rate=False):
    from autosing.fc_model import Wav2Vec2ForFrameClassification

    if model_name not in models:
        models[model_name] = Wav2Vec2ForFrameClassification.from_pretrained(model_name).to(device)
    model = models[model_name]

    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform[:1]

    target_sample_rate = 16000
    waveform = F.resample(waveform, sample_rate, target_sample_rate)
    sample_rate = target_sample_rate

    chunk_length = 30 * sample_rate
    num_chunks = int(math.ceil(waveform.shape[-1] / chunk_length))

    with torch.inference_mode():
        emission = torch.cat([
            torch.log(torch.softmax(model(chunk.to(device)).logits, dim=-1))
            for chunk in torch.chunk(waveform, num_chunks, dim=-1)
        ], 1)

    if not ret_frame_rate:
        return emission, waveform, sample_rate
    else:
        return (
            emission, waveform, sample_rate,
            round(emission.shape[-2] / (waveform.shape[-1] / sample_rate))
        )

charsiu_processor = None

# TODO: Get rid of this dependency
def get_charsiu_processor():
    global charsiu_processor
    if charsiu_processor is None:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        # This is expected to be done like this by the library :)
        sys.path.append(str(pathlib.Path(__file__).parent / '../charsiu/src/'))
        from processors import CharsiuPreprocessor_en

        charsiu_processor = CharsiuPreprocessor_en()
    return charsiu_processor

def text2tokens(text):
    if text == "": return [], [], []
    phones, text = get_charsiu_processor().get_phones_and_words(text)
    if len(phones) == 0: return [], [], []
    tokens = charsiu_processor.get_phone_ids(phones)[1:-1]
    return tokens, text, [len(ph) for ph in phones]

def compute_silence(emission, min_repeat=100, mult_threshold=200):
    sil_idx = 0

    p = torch.softmax(emission[0], dim=-1)
    srted, indices = p.sort(-1)
    mask = (srted[:, -1] > mult_threshold * srted[:, -2]) & (indices[:, -1] == sil_idx)

    uniq, inverse, counts = torch.unique_consecutive(
        mask,
        return_inverse=True,
        return_counts=True,
    )

    mask = uniq & (counts > min_repeat)
    return mask[inverse]

def compute_va(audio_file, model_name=None, return_prob=False, **kwargs):
    emission, _, _, frame_rate = compute_emission(
        audio_file,
        model_name=model_name,
        ret_frame_rate=True
    )

    if return_prob:
        return torch.softmax(emission[0], dim=-1)[:, 1:].sum(1)

    mask = compute_silence(emission, **kwargs)
    mask = mask[:mask.shape[-1] // frame_rate * frame_rate]
    return ~mask.reshape(-1, frame_rate).any(-1).cpu()

lrc_regex = r"^\[(\d\d):(\d\d\.\d\d)\]\s?(.*)$"
def parse_lrc(lrc_string):
    matches = re.finditer(lrc_regex, lrc_string, re.MULTILINE)

    times = []
    lines = []
    for match in matches:
        m, s, text = match.groups()
        s = float(s) + 60 * int(m)
        times.append(s)
        lines.append(text)

    return times, lines

class AlignException(Exception):
    ...

def compute_alignment(
    audio_file,
    lrc_text=None,
    raw_text=None,
    model_name=None,
    ret_silence=False,
):
    if lrc_text is not None:
        lrc_times, raw_lines = parse_lrc(lrc_text)
    else:
        tokens, lines, word_lengths = [[x] for x in text2tokens(raw_text)]
        raw_lines = raw_text.split("\n")
        lrc_times = None

    tokens, lines, word_lengths = [list(el) for el in zip(*[
        text2tokens(line)
        for line in raw_lines
    ])]

    emission, waveform, sample_rate = compute_emission(audio_file, model_name)
    frame_length = 1 / (sample_rate * (emission.shape[1] / waveform.shape[1]))

    def align(emission, tokens):
        for repeat in [1, 2, 4]:
            try:
                repeated = torch.repeat_interleave(emission, repeat, dim=-2)
                targets = torch.tensor([tokens], dtype=torch.int32, device=device)
                alignments, scores = F.forced_align(repeated, targets, blank=0)
                token_spans = F.merge_tokens(alignments[0], scores[0])

                for span in token_spans:
                    span.start /= repeat
                    span.end /= repeat

                break
            except Exception as e:
                print(e)
                continue
        else:
            raise AlignException(
                f"Too many tokens ({len(tokens)}) for emission (length {emission.shape[-2]})"
            )

        return token_spans

    def from_token_spans(spans, inlier_ratio=None, lrclib_offset=None):
        starts, ends, scores = [list(el) for el in zip(*[
            (frame_length * span.start, frame_length * span.end, span.score)
            for span in spans
        ])]

        return Alignment(
            np.array(starts),
            np.array(ends),
            np.array(scores),
            np.array(sum(tokens, start=[])),
            sum(lines, start=[]),
            raw_lines,
            np.array(sum(word_lengths, start=[])),
            np.array([len(line) for line in lines]),
            inlier_ratio,
            lrclib_offset,
        )

    token_spans = align(emission[:1], sum(tokens, start=[]))

    global_alignment = from_token_spans(token_spans)

    if lrc_times is None:
        return None, global_alignment

    lrc_starts = torch.tensor(lrc_times)
    line_starts, _ = global_alignment.line_spans
    line_starts = torch.tensor(line_starts)

    max_inliers = 0
    best_offset = None
    for i, (lrc_time, our_time) in enumerate(zip(lrc_starts, line_starts)):
        if math.isnan(our_time): continue
        offset = lrc_time - our_time
        mask = (lrc_starts - offset - line_starts).abs() < 1
        num_inliers = mask[~line_starts.isnan()].sum().item()
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            inlier_ratio = num_inliers / (~line_starts.isnan()).sum().item()
            best_offset = (lrc_starts - line_starts)[mask].mean().item()

    audio_length = sample_rate * waveform.shape[1]

    constr_spans = []

    line_iter = zip(tokens, lrc_times, lrc_times[1:] + [audio_length])
    for line_tokens, lrc_start, lrc_end in line_iter:
        if not line_tokens: continue

        def s2frame(t): return int(round((t - best_offset) / frame_length))

        start_frame, end_frame = s2frame(lrc_start), s2frame(lrc_end)
        start_frame = max(0, start_frame - 1)
        end_frame = max(0, end_frame + 1)

        token_spans = align(
            emission[:1, start_frame:end_frame],
            line_tokens
        )

        for span in token_spans:
            span.start += start_frame
            span.end += start_frame

        constr_spans += token_spans 

    constr_alignment = from_token_spans(
        constr_spans,
        inlier_ratio=inlier_ratio,
        lrclib_offset=best_offset
    )

    if ret_silence:
        return (
            constr_alignment,
            global_alignment,
            compute_silence(emission).cpu(),
        )
    else:
        return constr_alignment, global_alignment

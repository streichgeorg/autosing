# AutoSing

## Samples
https://streichgeorg.github.io/autosing_samples/

## Setup

The code was developed under Python 3.11 and expects PyTorch 2.1.2 (including torchaudio, torchvision) to be installed. To get started, clone the repository and run `pip install .`

## Data Processing

Most of the processing is implemented in `processing.py`. Data and intermediate results are stored in Parquet files. The initial data is expected to be split across multiple Parquet partitions using the following directory structure:

```
<dataset>/
    0/index.parquet
    1/index.parquet
    2/index.parquet
    ...
```

Each `index.parquet` file should contain the following columns:

- `id`: Unique identifier for each song.
- `audio`: Byte string of audio data encoded in some format supported by PyTorch.

Optional fields:

- `lrc_lyrics`: Lyrics in LRC format, the code will still compute a word-level alignment but will base it on the LRC timestamps.
- `raw_lyrics`: Text-only lyrics.
- `artist`: Artist name, used to train the embedding model.

After running the necessary processing steps, the `construct_dataset.py` script is used to shuffle and chunk the dataset.

## Training

Training runs can be started as follows:

### Text-to-Semantic

```
python3 autosing/train.py --task_name t2s \
--task_args '{"size": "<model size: small, medium, large>"}' \
--dataset-config '{"path": "<path to your dataset>"}' \
--tunables '{"lr0": 4e-3}'
```

### Semantic-to-Audio

```
python3 autosing/train.py --task_name sm2a \
--task_args '{"size": "<model size: small, medium, large>"}' \
--dataset-config '{"path": "<path to your dataset>", "multiscale": <whether to enable multiscale training>}' \
--tunables '{"lr0": 3e-3}'
```

## Inference

Samples can be generated using the `sing.py` script like this. The script expects two lines of lyrics, the first (second) line controls the first (last) 15 seconds of output.

```
python3 autosing/sing.py \
--reference <reference song to take the instrumentals and artist embedding from> \
--sm2a-model <trained sm2a model> --t2s-model <trained t2s model> \
--lyrics $'la la la la\nla la la la'
```

## Acknowledgments

Our architecture and training code is based on the wonderful [WhisperSpeech](https://github.com/collabora/WhisperSpeech) codebase.


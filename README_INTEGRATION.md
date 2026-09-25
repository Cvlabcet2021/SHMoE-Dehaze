# SHMoE-Dehaze implementation

This package contains the remaining modules for the supplied
SHMoE-Dehaze configuration and the supplied HazeEncoder/SAM3Encoder.

## Important integration point

`SAM3Encoder` is an adapter. The project does not guess the official
SAM-3 checkpoint loading API. In `train/train_shmoe.py` and
`inference/test.py`, replace:

    sam3_model = None

with the official SAM-3 model construction/loading code.

The rest of the network expects the SAM-3 encoder to provide a
256-channel spatial feature map.

## HazeEncoder compatibility

The supplied HazeEncoder returns 128 channels. `models/routing.py`
therefore projects 128 -> 256 before semantic-haze cross attention.

## Dataset API

The training scripts assume the dataset returns either:

    (hazy, gt)

or a dictionary containing:

    {"hazy": ..., "gt": ...}

Adjust the dataset constructor in the training scripts if your
`datasets/reside_its.py` uses a different signature.

## Quick syntax check

From the project root:

    python -m compileall models losses train inference utils

## Important research note

The diffusion module is a compact implementation of the paper-level
DDPM/DDIM idea and the specified conditioning rails. It is not a
drop-in reproduction of an undisclosed official HazeDiff-CPG codebase.
The official SAM-3 loading/API and any exact paper-specific dataset
loader details must be connected before claiming exact reproduction.

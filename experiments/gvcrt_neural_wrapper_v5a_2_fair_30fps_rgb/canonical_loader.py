"""The single, validated RGB PNG loader used for both datasets."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def canonical_arrays(sequence_dir):
    directory = Path(sequence_dir)
    expected = [directory / f'frame_{i:06d}.png' for i in range(64)]
    assert sorted(directory.glob('*.png')) == expected, f'Expected exactly 64 PNGs: {directory}'
    for path in expected:
        with path.open('rb') as handle:
            header = handle.read(26)
        assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[24:26] == bytes([8, 2]), path
        with Image.open(path) as image:
            assert image.mode == 'RGB' and image.size == (1920, 1080), path
            array = np.array(image, dtype=np.uint8, copy=True)
        yield array

def load_canonical_frames(sequence_dir):
    import torch
    return [torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            for a in canonical_arrays(sequence_dir)]

def rgb_hash(sequence_dir):
    digest = hashlib.sha256()
    for array in canonical_arrays(sequence_dir):
        digest.update(array.tobytes())
    return digest.hexdigest()


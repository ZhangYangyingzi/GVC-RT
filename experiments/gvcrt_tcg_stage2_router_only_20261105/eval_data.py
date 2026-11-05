"""Read the exact existing RGB windows; no resampling or source mutation."""
from common_s2 import *
import hashlib


def read_frames(video):
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    arrays = []
    digest = hashlib.sha256()
    if video['domain'] in ('ulong', 'ulong_new'):
        source = Path(video['source_path'])
        assert sha(source) == video['source_sha256'], source
        capture = cv2.VideoCapture(str(source))
        indices = video['source_frame_indices']
        assert indices == list(range(indices[0], indices[0]+64))
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        try:
            for _ in range(64):
                ok, bgr = capture.read()
                assert ok, video['sequence']
                arrays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    else:
        for i in range(64):
            path = Path(video['input_dir'])/(f'im{i+1:05d}.png' if video['domain']=='hevc_b' else f'frame_{i:06d}.png')
            assert sha(path) == video['frame_file_sha256'][i]
            with Image.open(path) as im:
                assert im.mode == 'RGB'
                arrays.append(np.asarray(im, dtype=np.uint8).copy())
    for i, array in enumerate(arrays):
        assert array.shape == (video['height'], video['width'], 3)
        raw = array.tobytes()
        assert hashlib.sha256(raw).hexdigest() == video['frame_rgb_sha256'][i], (video['sequence'], i)
        digest.update(raw)
    # The old ULong manifest hashes the concatenated frame-hash strings;
    # the other manifests hash concatenated RGB bytes. Individual frames
    # were verified above in both cases.
    observed = hashlib.sha256(''.join(video['frame_rgb_sha256']).encode()).hexdigest() if video['domain']=='ulong' else digest.hexdigest()
    assert observed == video['rgb_sha256']
    return arrays


def verify_video(video):
    arrays = read_frames(video)
    return dict(sequence=video['sequence'], domain=video['domain'], split=video['split'], frames=len(arrays),
                rgb_sha256=video['rgb_sha256'], status='PASS')

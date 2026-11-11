"""Read audited real frames, one identical spatial crop for the whole window."""
from common_s2 import *


def read_arrays(row, training):
    if row.get('long_video'):
        import cv2
        assert sha(row['source_path'])==row['source_sha256']
        indices=row['source_frame_indices'];assert indices==list(range(128))
        cap=cv2.VideoCapture(row['source_path']);arrays=[]
        try:
            for i in indices:
                ok,bgr=cap.read();assert ok
                arrays.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        finally:cap.release()
        return arrays
    if not training:
        from eval_data import read_frames
        return read_frames(row)
    if row['domain']=='ulong':
        from data_reader import read_arrays as read
        return read(row)
    from PIL import Image
    import numpy as np
    arrays=[];file_hashes=[];rgb_hashes=[]
    assert row['domain']=='vimeo' and row['source_frame_indices']==list(range(7))
    assert row['video'] in set((ROOT/'inputs/sep_trainlist.txt').read_text().split())
    assert row['video'] not in set((ROOT/'inputs/sep_testlist.txt').read_text().split())
    for i in range(7):
        path=Path(row['source_path'])/f'im{i+1}.png';file_hashes.append(sha(path))
        with Image.open(path) as image:
            assert image.mode=='RGB' and image.size==(448,256)
            a=np.array(image);rgb_hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
            arrays.append(a[:,row['crop_x']:row['crop_x']+256].copy())
    receipt=dict(status='PASS',sample_key=row['sample_key'],files=file_hashes,RGB=rgb_hashes,source_frame_indices=list(range(7)),real_frames=7,repeat=False)
    path=ROOT/'audits/window_reads'/f'{row["sample_key"]}.json'
    if path.exists():assert load(path)==receipt
    else:dump(path,receipt)
    return arrays

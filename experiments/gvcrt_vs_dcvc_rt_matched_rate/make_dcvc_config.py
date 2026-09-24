#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
man=json.loads((ROOT/'manifest.json').read_text())
for v in man['videos']:
 tag=f"{v['dataset']}_{int(v['video_id']):02d}"
 cfg={'root_path':str(ROOT/'source_frames'),'test_classes':{'matched':{'test':1,'base_path':'','src_type':'png','sequences':{tag:{'height':1080,'width':1920,'intra_period':-1,'frames':64}}}}}
 p=ROOT/'parts'/f'{tag}_dcvc_config.json';p.write_text(json.dumps(cfg,indent=2)+'\n')

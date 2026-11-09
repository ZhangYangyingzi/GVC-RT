"""Translate the task's seven-panel layout for FFmpeg without xstack fill.

Only visualization filter syntax changes. Video decoding, model inference,
metrics, checkpoints and all frozen experiment sources remain untouched.
"""
from pathlib import Path
import sys
ROOT_PATH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_PATH))
from common_s2 import *
import re

def rewrite(arguments):
    arguments=list(arguments)
    if '-filter_complex' not in arguments:return arguments,False
    i=arguments.index('-filter_complex')+1;f=arguments[i]
    if ':fill=black' not in f:return arguments,False
    pattern=r'\[v0\]\[v1\]\[v2\]\[v3\]\[v4\]\[v5\]\[v6\]xstack=inputs=7:layout=0_0\|(\d+)_0\|(\d+)_0\|(\d+)_0\|0_(\d+)\|(\d+)_(\d+)\|(\d+)_(\d+):fill=black\[out\]'
    found=re.search(pattern,f)
    assert found,'UNRECOGNIZED_XSTACK_LAYOUT_NO_SILENT_REWRITE'
    a,b,c,h,d,e,g,j=map(int,found.groups())
    assert b==2*a and c==3*a and d==a and g==2*a and e==h and j==h
    replacement=('[v0]split=2[compat_first][compat_blank_source];'
        '[compat_blank_source]lutrgb=r=0:g=0:b=0[compat_blank];'
        '[compat_first][v1][v2][v3]hstack=inputs=4[compat_top];'
        '[v4][v5][v6][compat_blank]hstack=inputs=4[compat_bottom];'
        '[compat_top][compat_bottom]vstack=inputs=2[out]')
    arguments[i]=f[:found.start()]+replacement+f[found.end():]
    assert ':fill=black' not in arguments[i]
    return arguments,True

def main():
    config=load(ROOT/'ops/ffmpeg_compat_config.json');real=Path(config['real_ffmpeg'])
    assert sha(real)==config['real_ffmpeg_SHA256']
    arguments,changed=rewrite(sys.argv[1:])
    if changed:
        env();record=dict(time=time.time(),PID=os.getpid(),wrapper_SHA256=sha(Path(__file__)),original_filter_SHA256=digest(sys.argv[1:]),translated_arguments_SHA256=digest(arguments),
            reason='Existing FFmpeg xstack has no fill option; same seven panels with an explicit black eighth tile via hstack/vstack',output=arguments[-1])
        with lock('ops/ffmpeg_compat_audit.lock'):
            p=ROOT/'audits/ffmpeg_compat_invocations.json'
            previous=load(p) if p.exists() else [];previous.append(record);dump(p,previous)
    os.execv(str(real),[str(real),*arguments])

if __name__=='__main__':main()

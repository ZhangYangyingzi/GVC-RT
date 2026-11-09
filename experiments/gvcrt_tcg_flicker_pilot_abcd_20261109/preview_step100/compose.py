"""Resume the CPU-only composition of seven already verified viewing clips."""
from pathlib import Path
import sys
ROOT_PATH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_PATH))
from common_s2 import *

def main():
    env();out=ROOT/'preview_step100'
    with lock('locks/quick_preview_100.lock'):
        prior=load(out/'status.json')
        if prior.get('status')=='FAILED':dump(out/'audits/initial_composition_failure.json',prior)
        dump(out/'status.json',dict(status='RUNNING',current_stage='CPU_composition',PID=os.getpid(),time=time.time()))
        source=module('preview_original_export',out/'export.py')
        compat=module('preview_ffmpeg_compat',ROOT/'ops/ffmpeg_compat.py')
        config=load(ROOT/'ops/ffmpeg_compat_config.json');assert sha(config['real_ffmpeg'])==config['real_ffmpeg_SHA256']
        def command(arguments):
            assert arguments[0]=='ffmpeg'
            changed,translated=compat.rewrite(arguments[1:])
            if '-filter_complex' in changed:
                assert translated
                i=changed.index('-filter_complex')+1
                assert changed[i].endswith('[out]')
                changed[i]=changed[i][:-5]+'[RGB_composite];[RGB_composite]scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p[out]'
                changed=changed[:-1]+['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',*changed[-1:]]
            subprocess.run([config['real_ffmpeg'],*changed],check=True,stdin=subprocess.DEVNULL)
        source.command=command
        entries=[load(out/'manifests'/f'{g}.json') for g in source.MODELS]
        for record in entries:
            assert record['status']=='PASS' and record['frames']==64
            assert sha(record['lossless_path'])==record['lossless_SHA256']
            assert sha(record['preview_path'])==record['preview_SHA256']
            assert record['identity']['preview_script_SHA256']==sha(out/'export.py')
        profile=load(ROOT/'manifests/quick_development.json')[0]
        from cache_codec import cache_paths
        key,_,receipt=cache_paths(profile['row'],5,False);proof=load(receipt)
        assert all(e['identity']['source_cache_identity']==key and e['identity']['stream_SHA256']==proof['stream_SHA256'] for e in entries)
        comparisons=[source.combined(entries,proof,profile['sequence'],crop) for crop in [False,True]]
        for c in comparisons:
            result=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,color_space,color_range','-of','json',c['path']],text=True)
            record=json.loads(result)['streams'][0];assert int(record['nb_read_frames'])==64 and record['color_space']=='bt709' and record['color_range']=='tv'
            c['ffprobe']=record
        from freeze import verify
        assert verify()
        manifest=dict(status='COMPLETE',purpose='Viewing-only intermediate checkpoint preview; not a metric evaluation or checkpoint selection',sequence=profile['sequence'],
            GT_only_preselected_category=profile['GT_category_rule'],external_QP=5,frames=64,checkpoint_step=100,models=entries,comparisons=comparisons,training_updates=0,frozen_protocol_unchanged=True,
            source_script_SHA256=sha(out/'export.py'),composition_script_SHA256=sha(Path(__file__)),FFmpeg_compat_script_SHA256=sha(ROOT/'ops/ffmpeg_compat.py'),time=time.time())
        dump(out/'manifest.json',manifest)
        dump(out/'status.json',dict(status='COMPLETE',PID=os.getpid(),models=7,frames=64,manifest=str(out/'manifest.json'),training_updates=0,time=time.time()))

if __name__=='__main__':main()

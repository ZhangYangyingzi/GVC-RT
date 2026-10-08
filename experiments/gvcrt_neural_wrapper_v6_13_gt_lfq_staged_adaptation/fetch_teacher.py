"""Archive official teacher definitions and provenance before any training."""
import argparse, subprocess
from io_utils import *
BASE='https://api.github.com/repos/TencentARC/SEED-Voken'
def fetch(url,path):
    path=Path(path);assert path.is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_name(path.name+'.download.tmp')
        for attempt in range(4):
            argv=['curl','-L','--fail','--connect-timeout','30','--max-time','300','--speed-time','45','--speed-limit','1024','--continue-at','-','-o',str(tmp),url]
            command(argv)
            with (ROOT/'logs/downloads.log').open('a') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:break
            dump(ROOT/'logs/failures'/f'download_{time.time_ns()}.json',dict(url=url,returncode=result.returncode,partial_path=str(tmp),partial_bytes=tmp.stat().st_size if tmp.exists() else 0,resume=True,attempt=attempt+1))
        result.check_returncode();tmp.replace(path)
    return dict(url=url,path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',action='store_true');parser.add_argument('--pretrain',action='store_true');args=parser.parse_args()
    command([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
    records=[];out=ROOT/'teacher_sources'
    records.append(fetch(BASE+'/commits/main',out/'official_commit.json'))
    commit=load(out/'official_commit.json')['sha']
    records.append(fetch(BASE+'/git/trees/'+commit+'?recursive=1',out/'official_tree.json'))
    paths=[x['path'] for x in load(out/'official_tree.json')['tree'] if x['type']=='blob']
    selected=[x for x in paths if x in ('docs/Open-MAGVIT2.md','LICENSE','src/Open_MAGVIT2/models/lfqgan.py','src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py','src/Open_MAGVIT2/modules/vqvae/lookup_free_quantize.py','src/Open_MAGVIT2/modules/ema.py','src/Open_MAGVIT2/data/base.py','src/Open_MAGVIT2/data/imagenet.py','scripts/evaluation/evaluation_256.sh') or ('configs' in x and 'Open-MAGVIT2' in x and x.endswith(('.yaml','.yml')))]
    for path in selected:records.append(fetch('https://raw.githubusercontent.com/TencentARC/SEED-Voken/'+commit+'/'+path,out/path))
    records.append(fetch('https://arxiv.org/html/2608.04891v1',out/'gvcrt_paper_v1.html'))
    records.append(fetch('https://api.github.com/repos/semcomm/GVC-RT/commits/main',out/'gvcrt_official_commit.json'))
    native_commit=load(out/'gvcrt_official_commit.json')['sha']
    records.append(fetch('https://api.github.com/repos/semcomm/GVC-RT/git/trees/'+native_commit+'?recursive=1',out/'gvcrt_official_tree.json'))
    records.append(fetch('https://raw.githubusercontent.com/semcomm/GVC-RT/'+native_commit+'/README.md',out/'gvcrt_official_README.md'))
    dump(ROOT/'audits/official_source_downloads.json',dict(commit=commit,records=records))
    if args.pretrain:
        records.append(fetch('https://raw.githubusercontent.com/TencentARC/SEED-Voken/'+commit+'/src/Open_MAGVIT2/models/lfqgan_pretrain.py',out/'src/Open_MAGVIT2/models/lfqgan_pretrain.py'))
        repo='TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain'
        fetch('https://huggingface.co/api/models/'+repo,out/'pretrain_hf_model.json');revision=load(out/'pretrain_hf_model.json')['sha']
        fetch('https://huggingface.co/api/models/'+repo+'/tree/'+revision,out/'pretrain_hf_files.json')
        fetch('https://huggingface.co/'+repo+'/resolve/'+revision+'/README.md',out/'pretrain_hf_README.md')
        record=fetch('https://huggingface.co/'+repo+'/resolve/'+revision+'/pretrain256_262144.ckpt',ROOT/'teacher_weights/pretrain256_262144.ckpt')
        expected=next(x for x in load(out/'pretrain_hf_files.json') if x['path']=='pretrain256_262144.ckpt')
        assert record['sha256']==expected['lfs']['oid'] and record['bytes']==expected['size']
        dump(ROOT/'audits/pretrain_teacher_checkpoint_source.json',dict(status='VERIFIED_DOWNLOAD',repo=repo,revision=revision,official_code_commit=commit,checkpoint=record,official_lfs_sha256=expected['lfs']['oid'],candidate_only=True))
        dump(ROOT/'audits/official_source_downloads.json',dict(commit=commit,records=records))
    if args.weights:
        repo='TencentARC/Open-MAGVIT2-Tokenizer-256-resolution'
        records.append(fetch('https://huggingface.co/api/models/'+repo,out/'hf_model.json'))
        revision=load(out/'hf_model.json')['sha']
        records.append(fetch('https://huggingface.co/api/models/'+repo+'/tree/'+revision,out/'hf_files.json'))
        records.append(fetch('https://huggingface.co/'+repo+'/resolve/'+revision+'/README.md',out/'hf_README.md'))
        record=fetch('https://huggingface.co/'+repo+'/resolve/'+revision+'/imagenet_256_L.ckpt',ROOT/'teacher_weights/imagenet_256_L.ckpt')
        expected=next(x for x in load(out/'hf_files.json') if x['path']=='imagenet_256_L.ckpt')
        assert record['sha256']==expected['lfs']['oid'] and record['bytes']==expected['size']
        dump(ROOT/'audits/teacher_checkpoint_source.json',dict(status='VERIFIED_DOWNLOAD',repo=repo,revision=revision,official_code_commit=commit,checkpoint=record,official_lfs_sha256=expected['lfs']['oid'],candidate_only=True))
    print('OFFICIAL SOURCES SAVED',commit,selected,flush=True)
if __name__=='__main__':main()

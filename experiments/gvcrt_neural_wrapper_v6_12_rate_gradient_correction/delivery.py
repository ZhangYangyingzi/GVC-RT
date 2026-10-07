"""Canonical uploadable frame/transition tables; invoked before final PASS."""
from v68_io import *
def package(raw,held):
    tables=[]
    for record in raw+held:
        d=record['dataset'];m=record['method'];i=record['video_index'];q=record['QP']
        base=point(d,m,i,q) if d in DATASETS else ROOT/'parts/heldout'/m/f'video_{i:03d}_qp{q}.json'
        for key,suffix in [('frame_metrics','.frames.csv'),('transitions','.transitions.csv')]:
            src=Path(record[key+'_path']);assert sha(src)==record[key+'_sha256']
            dst=base.with_suffix(suffix)
            if dst.exists():assert sha(dst)==record[key+'_sha256']
            else:dst.write_bytes(src.read_bytes())
            tables.append(dict(path=str(dst.relative_to(ROOT)),source=str(src),sha256=sha(dst)))
    assert len(tables)==4872
    dump(ROOT/'audits/raw_table_delivery.json',dict(status='PASS',tables=tables))
    local={'features','heldout_features','visualizations','bitstreams','heldout_bitstreams'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts)&local) and (p.suffix in ('.py','.json','.csv','.jsonl','.log') or p.name=='.gitignore')]
    result=subprocess.run(['git','check-ignore','--stdin'],cwd=REPO,input='\n'.join(str(p.relative_to(REPO)) for p in files)+'\n',capture_output=True,text=True)
    assert result.returncode==1 and not result.stdout,result.stdout
    dump(ROOT/'audits/delivery_upload_audit.json',dict(status='PASS',checked=len(files),ignored=[]))

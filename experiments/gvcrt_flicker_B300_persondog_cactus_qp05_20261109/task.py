"""Fixed inference-only task identities and immutable-input verification."""
from common_s2 import *

def tasks():
    return [dict(index=i,profile=p,QP=q,rank=i) for i,(p,q) in enumerate((p,q) for p in load(ROOT/'profiles.json') for q in [0,5])]

def verify():
    m=load(ROOT/'source_manifest.json');assert m['status']=='FROZEN'
    assert sha(ROOT/'protocol.json')==m['protocol_SHA256'] and sha(ROOT/'profiles.json')==m['profiles_SHA256'] and sha(ROOT/'rois.json')==m['rois_SHA256']
    for name,value in m['code'].items():assert sha(ROOT/name)==value,('TASK_CODE_CHANGED',name)
    for name,value in m['protected_sources'].items():assert sha(name)==value,('READONLY_SOURCE_CHANGED',name)
    assert sha(B_FINAL)==B_FINAL_SHA
    return True

def point_path(profile,q,model):
    return ROOT/f'long_evaluation/{model}/step_{300 if model=="B" else 0:04d}/points/{profile["sequence"]}_qp{q}.json'

def completed_points():
    return sum(point_path(t['profile'],t['QP'],model).exists() for t in tasks() for model in ['Original','B'])

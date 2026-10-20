from io32 import *
def main():
 rows=[x['source_row'] for x in load(ROOT/'plans/shared.json')['updates'][:500]];paths={r['source_path']:r['source_sha256'] for r in rows}
 for vs in load(ROOT/'eval_manifest.json').values():
  for v in vs:paths[v.get('source_path',v.get('path'))]=v['source_sha256']
 proof=[]
 for p,expected in paths.items():
  actual=sha(p);assert actual==expected,(p,actual,expected);proof.append(dict(path=p,sha256=actual,bytes=Path(p).stat().st_size))
 d=load(ROOT/'audits/data_isolation.json');d['current_input_source_hashes']=proof;d['all_current_source_hashes_verified']=True;dump(ROOT/'audits/data_isolation.json',d)
 print('INPUT SOURCE HASH PASS',len(proof),flush=True)
if __name__=='__main__':main()

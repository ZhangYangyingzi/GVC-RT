from common_s2 import *
def main():
 env();protocol=load(ROOT/'protocol.json');assert protocol['status']=='FROZEN'
 files=['execution_policy.py','common_s2.py','b300_loader.py','model_context.py','experiment_model.py','convlora.py','fp32_island.py','loss.py','train.py','calibrate.py','preflight.py','temporal.py','cache_codec.py','data_reader.py','inherited_reader.py','eval_data.py','original_context.py','original_runtime.py','quality.py','evaluation.py','selection.py']
 code={n:sha(ROOT/n) for n in files}
 for tree in ['runtime_tree','reference_tree','vendor']:
  for p in (ROOT/tree).rglob('*.py'):code[str(p.relative_to(ROOT))]=sha(p)
 record=dict(status='FROZEN',base_commit=SOURCE_COMMIT,protocol_sha256=sha(ROOT/'protocol.json'),plan_index_sha256=sha(ROOT/'plans/index.json'),code=code,protected_sources=load(ROOT/'source_inputs.json')['readonly'],immutable_manifests={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'manifests').glob('*.json')},time=time.time())
 if (ROOT/'source_manifest.json').exists():verify_sources();return
 dump(ROOT/'source_manifest.json',record)
if __name__=='__main__':main()

from util import *
setup_torch()
from model import load_cref,receiver_named,frozen_hashes,G11
c=load_cref();names=receiver_named(c);b=G11().cuda();r=dict(status='PASS',device=device_info(),native_parameters=sum(p.numel() for _,p in names),native_names=[n for n,p in names],new_parameters=sum(p.numel() for p in b.parameters()),C_ref_B300_composite=thash(c['pm'].state_dict().items()),S_parameter_sha256=thash(c['S'].state_dict().items()),frozen=frozen_hashes(c),modules=str(b));dump(ROOT/'audits/runtime_inspection.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['modules','frozen','native_names']},indent=2))

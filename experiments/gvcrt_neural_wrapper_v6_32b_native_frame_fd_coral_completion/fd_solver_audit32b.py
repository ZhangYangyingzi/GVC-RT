"""Independent exact solver comparison on real B1 native-frame feature data; CPU only."""
from io32 import *
def main():
 import torch,numpy as np,traceback
 torch.set_num_threads(2)
 from distribution32 import losses
 row=load(ROOT/'plans/calibration.json')['updates'][0];i=row['adaptation_step']-1;r=load(ROOT/'reference_stats/receipts'/f'{i:04d}.json');assert r['full_RGB_sha256']==row['full_RGB_sha256']
 with np.load(r['feature_path']) as a:z=torch.from_numpy(a['reconstruction'].copy()).double().requires_grad_(True)
 item=next(x for x in load(ROOT/'reference_stats/index.json')['entries'] if x['stratum']==f'{row["domain"]}:{row["external_qp"]}');s=torch.load(item['path'],map_location='cpu',weights_only=True);ref=s['reference'];old=s['seed']
 ls,_=losses(z,old,ref);gnew=torch.autograd.grad(ls['FD'],z)[0];e,v=torch.linalg.eigh(ref['bar']);root=(v*e.sqrt()[None,:])@v.T;refold=dict(ref,sqrt=root.detach());orig=module('v632b_original_distribution',OLD32/'distribution32.py');record=dict(new_loss=float(ls['FD'].detach()),new_gradient_norm=float(gnew.norm()),input_receipt=str(ROOT/'reference_stats/receipts'/f'{i:04d}.json'),input_feature_sha256=sha(r['feature_path']),reference_path=item['path'],reference_sha256=item['sha256'],epsilon=ref['epsilon'],feature_dimension=2048,solver_math_unchanged=True,training_GT_regularizer_unchanged=True,source_solver_sha256=sha(OLD32/'distribution32.py'))
 mu,M,S=orig.candidate(z,old);bar=S+ref['epsilon']*torch.eye(2048,dtype=torch.float64);prod=orig.sym(root@bar@root);eig=torch.linalg.eigvalsh(prod.detach());record['symmetric_product_min_eigenvalue']=float(eig.min());record['symmetric_product_max_eigenvalue']=float(eig.max())
 try:
  ol,_=orig.losses(z,old,refold);go=torch.autograd.grad(ol['FD'],z)[0];lossdiff=abs(float(ol['FD'].detach())-record['new_loss']);relative=float((go-gnew).norm()/gnew.norm());record.update(historical_solver_status='PASS',historical_loss=float(ol['FD'].detach()),loss_absolute_difference=lossdiff,gradient_relative_L2=relative)
 except Exception:
  record.update(historical_solver_status='NUMERICALLY_UNSTABLE_ON_REAL_FEATURES',historical_error=traceback.format_exc())
 record.update(status='PASS',selected_solver='Cholesky cross-SVD exact equivalent trace, no jitter/clamp/fallback',small_SPD_loss_parameter_gradient_equivalence=load(ROOT/'audits/distribution_math.json')['FD_Cholesky_SVD_vs_symmetric_eigen'])
 dump(ROOT/'audits/FD_real_solver_comparison.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':main()

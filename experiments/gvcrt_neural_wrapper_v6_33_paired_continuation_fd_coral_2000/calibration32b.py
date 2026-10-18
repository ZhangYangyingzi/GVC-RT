"""Real crop and native full-chain gradient calibration, without optimizer/EMA updates."""
from io32 import *
import argparse,traceback,resource,math

def get_stratum(key,device):
 import torch
 from distribution32 import to
 item=next(r for r in load(ROOT/'reference_stats/index.json')['entries'] if r['stratum']==key);assert sha(item['path'])==item['sha256'];s=torch.load(item['path'],map_location='cpu',weights_only=True)
 return to(s['seed'],device),to(s['reference'],device)
def feature_derivatives(z,old,ref,objectives=('FD','mean','CORAL')):
 import torch
 from distribution32 import losses
 leaf=z.to('cuda:0').double().requires_grad_(True);ls,candidate=losses(leaf,old,ref);gr={}
 for i,name in enumerate(objectives):
  g=torch.autograd.grad(ls[name],leaf,retain_graph=i<len(objectives)-1)[0];assert g.isfinite().all() and g.norm()>0;gr[name]=g.detach()
 return gr,{k:float(v.detach()) for k,v in ls.items()},candidate

def run(index,gpu):
 os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
 import torch
 from training32 import setup,official_model,seed,rng_state,restore_rng,buffer_state,restore_buffers,crop_c1,full_pass,groups,norm,dot,check_frozen
 from frames32 import PlanFrames
 p=ROOT/'calibration/receipts'/f'{index:02d}.json';sig=signature('calibrate')
 if p.exists() and load(p)['signature']==sig:return
 ctx=setup();official=official_model();models=ctx[3];params=ctx[6];seed(load(ROOT/'config.json')['seed']);crng=rng_state();seed(load(ROOT/'config.json')['seed_global']);grng=rng_state();buf=buffer_state(models);initial={k:ctx[0].module_hash(m) for k,m in models.items()};oh=tensor_hash(official.state_dict());row=load(ROOT/'plans/calibration.json')['updates'][index];key=f'{row["domain"]}:{row["external_qp"]}';old,ref=get_stratum(key,'cuda:0');tic=time.time();torch.cuda.reset_peak_memory_stats()
 restore_rng(crng);f=PlanFrames(row,'cuda:0');g0,summary,trace=crop_c1(ctx,f,row['external_qp']);f.close();restore_buffers(models,buf)
 restore_rng(grng);f=PlanFrames(row,'cuda:0',full=True);z,gt,_,_=full_pass(ctx,f,row['external_qp'],official);f.close();restore_buffers(models,buf)
 fg,values,_=feature_derivatives(z,old,ref);restore_rng(grng);f=PlanFrames(row,'cuda:0',full=True);zr,_,gg,replay=full_pass(ctx,f,row['external_qp'],official,fg,expected=z);f.close();restore_buffers(models,buf);restore_rng(crng)
 n0=norm(g0);ns={k:norm(g) for k,g in gg.items()};assert n0>1e-12 and all(n>1e-12 and math.isfinite(n) for n in ns.values());assert [r['actual_qp'] for r in trace]==row['actual_qps']
 stats={k:dict(norm=ns[k],modules=groups(models,g),cosine_ref=dot(g0,g)/(n0*ns[k])) for k,g in gg.items()};assert all(v['norm']>0 and v['finite'] for s in stats.values() for v in s['modules'].values())
 check_frozen(ctx);assert initial=={k:ctx[0].module_hash(m) for k,m in models.items()} and oh==tensor_hash(official.state_dict());assert all(p.grad is None for p in params)
 dump(p,dict(status='PASS',signature=sig,index=index,plan_sha256=sha(ROOT/'plans/calibration.json'),shared_seed_sha256=sha(ROOT/'reference_stats/index.json'),row=row,gradient_ref=dict(norm=n0,modules=groups(models,g0)),gradients=stats,mean_CORAL_dot=dot(gg['mean'],gg['CORAL']),losses=values,crop_losses=summary,lambda_FD_candidate=.1*n0/ns['FD'],w_mu_candidate=n0/ns['mean'],w_cov_candidate=n0/ns['CORAL'],replay=replay,RGB_gradients_verified=True,optimizer_updates=0,EMA_commits=0,frozen_verified=True,model_unchanged=True,physical_GPU=gpu,seconds=time.time()-tic,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
 print('CALIBRATION PASS',index,time.time()-tic,flush=True)
def aggregate():
 import statistics
 rows=[load(ROOT/'calibration/receipts'/f'{i:02d}.json') for i in range(40)];assert all(r['status']=='PASS' and r['signature']==signature('calibrate') for r in rows)
 wmu=statistics.median(r['w_mu_candidate'] for r in rows);wcov=statistics.median(r['w_cov_candidate'] for r in rows);ks=[]
 for r in rows:
  n=math.sqrt(wmu*wmu*r['gradients']['mean']['norm']**2+wcov*wcov*r['gradients']['CORAL']['norm']**2+2*wmu*wcov*r['mean_CORAL_dot']);assert n>1e-12;ks.append(.1*r['gradient_ref']['norm']/n)
 k=statistics.median(ks);out=dict(status='PASS',completed_windows=40,expected_windows=40,lambda_FD=statistics.median(r['lambda_FD_candidate'] for r in rows),w_mu=wmu,w_cov=wcov,k=k,alpha_mu=k*wmu,alpha_cov=k*wcov,rho_cal=.1,epsilon=1e-12,calibration_signature=signature('calibrate'),plan_sha256=sha(ROOT/'plans/calibration.json'),shared_seed_sha256=sha(ROOT/'reference_stats/index.json'),receipts=[dict(path=str(ROOT/'calibration/receipts'/f'{i:02d}.json'),sha256=sha(ROOT/'calibration/receipts'/f'{i:02d}.json')) for i in range(40)],optimizer_updates=0,EMA_commits=0,training_only=True,frozen_after_calibration=True)
 dump(ROOT/'calibration/final.json',out);write(ROOT/'calibration/gradient_summary.csv',[dict(index=r['index'],domain=r['row']['domain'],QP=r['row']['external_qp'],reference_norm=r['gradient_ref']['norm'],**{f'{k}_norm':v['norm'] for k,v in r['gradients'].items()},lambda_FD_candidate=r['lambda_FD_candidate'],w_mu_candidate=r['w_mu_candidate'],w_cov_candidate=r['w_cov_candidate']) for r in rows])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int);ap.add_argument('--gpu',type=int,choices=(4,5,6,7),default=4);ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 try:aggregate() if a.aggregate else run(a.index,a.gpu)
 except Exception:error(dict(stage='calibration32b',arguments=vars(a),error=traceback.format_exc()));raise

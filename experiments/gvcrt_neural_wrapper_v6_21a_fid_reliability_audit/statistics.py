"""Exact sample-covariance FID and unbiased polynomial MMD; paired resampling."""
from io21 import *
import argparse,math,traceback

def numpy_fid(x,y):
 import numpy as np
 x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);a=x-x.mean(0);b=y-y.mean(0)
 return float(max(((x.mean(0)-y.mean(0))**2).sum()+(a*a).sum()/(len(x)-1)+(b*b).sum()/(len(y)-1)-2*np.linalg.svd(a@b.T/(len(x)-1),compute_uv=False).sum(),0))
class PairedEstimator:
 """Collapse repeated identical observations using counts, without changing the estimator."""
 def __init__(self,x,y,device):
  import torch
  self.t=torch;x=torch.as_tensor(x,dtype=torch.float64,device=device);y=torch.as_tensor(y,dtype=torch.float64,device=device);self.xx=x@x.T;self.yy=y@y.T;self.xy=x@y.T;dim=x.shape[1]
  self.kxx=(1+self.xx/dim)**3;self.kyy=(1+self.yy/dim)**3;self.kxy=(1+self.xy/dim)**3
 def measure(self,counts):
  t=self.t;w=t.as_tensor(counts,dtype=t.float64,device=self.xx.device);n=w.sum();u=t.nonzero(w>0).flatten();sw=w[u].sqrt();xxw=self.xx@w;yyw=self.yy@w;xyw=self.xy@w;wxy=w@self.xy
  vx=w@xxw;vy=w@yyw;vxy=w@xyw;mu=(vx+vy-2*vxy)/n**2;trace=(w@(self.xx.diagonal()+self.yy.diagonal())-(vx+vy)/n)/(n-1)
  cross=self.xy[u[:,None],u[None,:]]-xyw[u,None]/n-wxy[None,u]/n+vxy/n**2;cross=cross*sw[:,None]*sw[None,:]/(n-1)
  nuclear=t.linalg.svdvals(cross,driver='gesvd').sum() if self.xx.is_cuda else t.linalg.svdvals(cross).sum()
  fid=float(t.clamp(mu+trace-2*nuclear,min=0).item())
  kid=float(((w@self.kxx@w-w@self.kxx.diagonal()+w@self.kyy@w-w@self.kyy.diagonal())/(n*(n-1))-2*(w@self.kxy@w)/(n*n)).item())
  assert math.isfinite(fid) and math.isfinite(kid);return fid,kid

def counts_for(lengths,n,repeat,dataset,bootstrap):
 import numpy as np
 di=DATASETS.index(dataset);rng=np.random.default_rng(np.random.SeedSequence([20261010,di,int(n),int(repeat),int(bootstrap)]));out=[]
 if bootstrap:
  for length in lengths:out.extend(np.bincount(rng.integers(0,length,size=length),minlength=length))
 else:
  quotas=[n//len(lengths)]*len(lengths)
  for i in rng.permutation(len(lengths))[:n%len(lengths)]:quotas[int(i)]+=1
  while any(q>l for q,l in zip(quotas,lengths)):
   excess=0
   for i,l in enumerate(lengths):
    if quotas[i]>l:excess+=quotas[i]-l;quotas[i]=l
   for i in rng.permutation(len(lengths)):
    take=min(excess,lengths[int(i)]-quotas[int(i)]);quotas[int(i)]+=take;excess-=take
  for length,k in zip(lengths,quotas):
   w=np.zeros(length,np.int64);w[rng.choice(length,size=k,replace=False)]=1;out.extend(w)
 return np.asarray(out,np.int64)

def summary(values,prefix=''):
 import numpy as np
 a=np.asarray(values);return {prefix+k:float(v)for k,v in dict(mean=a.mean(),std=a.std(ddof=1)if len(a)>1 else 0,median=np.median(a),p2_5=np.percentile(a,2.5),p97_5=np.percentile(a,97.5)).items()}

def tests():
 import numpy as np,torch
 rng=np.random.default_rng(20261010);x=rng.normal(size=(32,12));y=rng.normal(size=(32,12));ids=rng.integers(0,32,32);counts=np.bincount(ids,minlength=32);est=PairedEstimator(x,y,'cpu');f,k=est.measure(counts);ref=numpy_fid(x[ids],y[ids]);assert abs(f-ref)<1e-10
 a=x[ids];b=y[ids];xx=(1+a@a.T/12)**3;yy=(1+b@b.T/12)**3;xy=(1+a@b.T/12)**3;kr=((xx.sum()-np.trace(xx))+(yy.sum()-np.trace(yy)))/(32*31)-2*xy.mean();assert abs(k-kr)<1e-12
 c=np.ones((128,12));f0,k0=PairedEstimator(c,c,'cpu').measure(np.ones(128));assert abs(k0)<1e-12
 vals=[]
 for _ in range(100):
  a=rng.normal(size=(128,12));b=rng.normal(size=(128,12));xx=(1+a@a.T/12)**3;yy=(1+b@b.T/12)**3;xy=(1+a@b.T/12)**3;vals.append((xx.sum()-np.trace(xx)+yy.sum()-np.trace(yy))/(128*127)-2*xy.mean())
 assert abs(np.mean(vals))<3*np.std(vals)/10
 dump(ROOT/'kid_implementation_integrity.json',dict(status='PASS',formula='sum offdiag Kxx/(m(m-1)) + sum offdiag Kyy/(n(n-1)) - 2 sum Kxy/(mn); K=(1+x dot y/d)^3',unbiased=True,identical_constant_features_KID=k0,independent_identically_distributed_synthetic_mean=float(np.mean(vals)),synthetic_std=float(np.std(vals)),duplicate_bootstrap_exact_enumeration_error=abs(k-kr),FID_collapsed_bootstrap_vs_explicit_error=abs(f-ref),unit_tests=4))
 print('ESTIMATOR TEST PASS',flush=True)

def reproduce():
 import numpy as np
 rows=[];deps=load(ROOT/'audits/dependencies.json');path=next(Path(p)for p in deps if Path(p).name=='fid_metric.py');fid=module('reproduction_historical_fid',path);records=[]
 for d in DATASETS:
  historical={(r['method'],int(r['QP'])):r for r in read(V20/'results'/d/'fid_pooled.csv')}
  gt=None
  for m in METHODS:
   for q in range(10):
    real=[];rec=[]
    for v in load(V20/'manifests'/f'{d}.json')['videos']:
     p=V20/'parts'/d/m/f'video_{v["video_index"]:02d}_qp{q}.json';r=load(p);assert r['source_rgb_sha256']==v['rgb_sha256'] and r['source_frame_indices']==v['source_frame_indices'] and sha(r['feature_path'])==r['feature_sha256'];records.append(dict(path=str(p),sha256=sha(p),features=r['feature_path'],feature_sha256=r['feature_sha256']))
     with np.load(r['feature_path'])as f:real.append(f['real']);rec.append(f['reconstruction'])
    x=np.concatenate(real);y=np.concatenate(rec)
    if gt is None:gt=x
    assert np.array_equal(gt,x);value=float(fid.low_rank_fid(gt,y));old=float(historical[m,q]['FID']);err=abs(value-old);passed=err<=1e-7+abs(old)*1e-9;assert passed,(d,m,q,old,value)
    rows.append(dict(dataset=d,method=m,QP=q,historical_value=old,reproduced_value=value,absolute_error=err,relative_error=err/max(abs(old),1e-30),status='PASS',samples=len(x)))
 write(ROOT/'results/v620_fid_reproduction.csv',rows);audit=dict(status='PASS',points=len(rows),tolerance='absolute 1e-7 + relative 1e-9',max_absolute_error=max(r['absolute_error']for r in rows),historical_FID_source=str(path),historical_FID_source_sha256=sha(path),source_records=records)
 dump(ROOT/'audits/v620_fid_reproduction.json',audit);dump(ROOT/'fid_reproduction_integrity.json',audit);print('V620 REPRODUCTION PASS',len(rows),flush=True)

def point_statistics(dataset,q,gpu):
 import numpy as np,torch
 os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);torch.set_num_threads(2);torch.manual_seed(20261010);torch.backends.cuda.matmul.allow_tf32=False
 frozen();out=ROOT/'statistics'/dataset/f'qp{q}.json'
 if out.exists():assert load(out)['status']=='PASS';return
 command([PYTHON,'-B','-u',str(Path(__file__).resolve()),'--dataset',dataset,'--qp',str(q),'--gpu',str(gpu)])
 gt=None;lengths=[];estimators={};rates={};features=[]
 for m in METHODS:
  real=[];rec=[];bps=[]
  for v in sources(dataset):
   r=load(point(dataset,m,v['video_index'],q));assert sha(r['feature_path'])==r['feature_sha256'];features.append(dict(path=r['feature_path'],sha256=r['feature_sha256']));bps.append(r['bpp'])
   with np.load(r['feature_path'])as f:real.append(f['real']);rec.append(f['reconstruction'])
  x=np.concatenate(real);y=np.concatenate(rec)
  if gt is None:gt=x;lengths=[len(a)for a in real]
  assert np.array_equal(gt,x);estimators[m]=PairedEstimator(gt,y,'cuda:0');rates[m]=float(np.mean(bps))
 n=len(gt);sample=[];boot=[];full=[];unavailable=[];t=time.time()
 for m,est in estimators.items():
  f,k=est.measure(np.ones(n));full.append(dict(dataset=dataset,method=m,QP=q,FID=f,KID=k,bpp=rates[m],samples=n))
 for size in [32,64,128,256,512,1024]:
  if size>n:unavailable.append(dict(sample_size=size,pool_size=n,reason='sample size exceeds available pool'));continue
  full_cached={m:(r['FID'],r['KID'])for m,r in zip(METHODS,full)}
  for repeat in range(100):
   counts=counts_for(lengths,size,repeat,dataset,False)
   for m,est in estimators.items():
    f,k=full_cached[m]if size==n else est.measure(counts);sample.append(dict(dataset=dataset,method=m,QP=q,sample_size=size,repeat=repeat,FID=f,actual_sample_count=int(counts.sum()),sequence_sampling_rule='equal sequence quotas; uniform without replacement inside sequence; paired identities across GT/method/QP'))
 for repeat in range(1000):
  counts=counts_for(lengths,n,repeat,dataset,True);values={m:est.measure(counts)for m,est in estimators.items()}
  for m,(f,k)in values.items():boot.append(dict(dataset=dataset,method=m,QP=q,repeat=repeat,FID=f,KID=k,actual_sample_count=n))
  if repeat%100==0:print('BOOTSTRAP',dataset,q,repeat,'elapsed',round(time.time()-t,1),flush=True)
 write(out.with_suffix('.sample.csv'),sample);write(out.with_suffix('.bootstrap.csv'),boot);write(out.with_suffix('.full.csv'),full)
 dump(out,dict(status='PASS',dataset=dataset,QP=q,pool_size=n,sequence_lengths=lengths,sample_rows=len(sample),bootstrap_rows=len(boot),bootstrap_repeats=1000,unavailable_sample_sizes=unavailable,features=features,elapsed_seconds=time.time()-t,physical_gpu=gpu,peak_memory_MiB=torch.cuda.max_memory_reserved()/2**20))
 frozen();print('STATISTICS PASS',dataset,q,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');ap.add_argument('--reproduce',action='store_true');ap.add_argument('--dataset');ap.add_argument('--qp',type=int);ap.add_argument('--gpu',type=int,choices=[4,5,6,7]);a=ap.parse_args()
 if a.test:tests()
 elif a.reproduce:reproduce()
 else:point_statistics(a.dataset,a.qp,a.gpu)
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'logs'/f'statistics_failure_{os.getpid()}.json',dict(error=traceback.format_exc()));raise

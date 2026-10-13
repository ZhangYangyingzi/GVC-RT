"""No optimizer steps: covariance invariants and frozen canary accounting."""
from audit31 import *
def main():
 import torch
 stat=module('v631_stat_selftest',V29/'stat_loss.py');torch.manual_seed(20261029)
 fs=[torch.randn(1,3,16,16),torch.randn(1,8,16,16),torch.randn(1,16,12,12),torch.randn(1,24,8,8),torch.randn(1,32,4,4)]
 same,_=stat.feature_loss(fs,fs);assert float(same)==0
 shifted=[f+torch.randn(1,f.shape[1],1,1) for f in fs];loss,_=stat.feature_loss(shifted,fs);assert float(loss)<=1e-10
 types={'color':210,'intervention':135,'probe':72};ts=tasklist();assert all(sum(t['stage']==st for t in ts)==n for st,n in types.items())
 dump(ROOT/'audits/selftest.json',dict(status='PASS',identity_covariance_loss=float(same),constant_feature_channel_shift_loss=float(loss),constant_feature_shift_tolerance=1e-10,annotation='centered covariance feature-space property; not RGB color invariance',temporary_updates=0,formal_updates=0,D_updates=0,canaries_count_as_formal_UIDs=True,optimizer_betas=load(ROOT/'config.json')['optimizer_effective']))
 print('SELFTEST PASS')
if __name__=='__main__':main()

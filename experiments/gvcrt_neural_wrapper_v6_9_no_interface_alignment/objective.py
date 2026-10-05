"""Exact existing L_B binder: no teacher, interface hooks, or alignment term."""
from v68_io import *
class Objective:
    def __init__(self,v4,im,pm,models,quality,cfg):
        sys.path.insert(0,str(V66))
        reference=module('v69_exact_B_objective',V66/'objective.py')
        self.fn=reference.bind(v4,im,pm,models['wrapper'],quality,cfg,
                               weight=cfg['lambda_struct'])
    def __call__(self,frames,q):return self.fn(frames,q)
    def close(self):pass

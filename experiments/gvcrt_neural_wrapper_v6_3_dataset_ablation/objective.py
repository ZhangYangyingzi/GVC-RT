"""Execute the exact AST of V6.2 run_clip, without copying or editing its math."""
import ast,math,types
from v63_io import *
def original_ast():
    tree=ast.parse((V62/'fullqp_train.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    return next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='run_clip')
def bind(v4,im,pm,wrapper,quality,cfg):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(V62));from fullqp_common import beta_q,lambda_q,frame_qp
    context=dict(v4=v4,im=im,pm=pm,wrapper=wrapper,quality=quality,cfg=cfg,torch=torch,F=F,math=math,
        args=types.SimpleNamespace(branch='schedule_s1p0'),beta_q=beta_q,lambda_q=lambda_q,frame_qp=frame_qp,load=load,ROOT=V62)
    tree=ast.Module(body=[original_ast()],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(V62/'fullqp_train.py'),'exec'),context)
    return context['run_clip']

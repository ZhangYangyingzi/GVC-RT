"""Mechanical source-delta audit; no model execution or optimizer updates."""
import ast
from io23 import *

def main():
    old=(V22/'window_training.py').read_text()
    expected=old.replace('from io21 import *','from io23 import *')
    expected=expected.replace("if rec_weight is None:rec_weight=load(ROOT/'calibration/lambda_rec.json')['lambda_rec'] if branch=='B1_reconstruction' else 0.", "if rec_weight is None:\n            rec_weight=load(ROOT/'config.json')['branch_lambda_rec'][branch]\n            assert rec_weight==WEIGHTS[branch]")
    expected=expected.replace("lambda_rec=(load(ROOT/'calibration/lambda_rec.json')['lambda_rec'] if mode=='train' and (ROOT/'calibration/lambda_rec.json').exists() else rec_weight)", "lambda_rec=rec_weight")
    expected=expected.replace("reconstruction_backward_scale=(rec_scale/rec_weight if rec_weight else rec_scale)", "reconstruction_backward_scale=rec_scale")
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse((ROOT/'window_training.py').read_text()))
    oldtrain=(V22/'train.py').read_text().replace('from io21 import *','from io23 import *').replace("'io21.py','gradient_tools.py'","'io23.py','gradient_tools.py'").replace("schema='v622'","schema='v623'")
    assert ast.dump(ast.parse(oldtrain))==ast.dump(ast.parse((ROOT/'train.py').read_text()))
    assert sha(ROOT/'plans/shared.json')==sha(V22/'plans/shared.json')
    dump(ROOT/'audits/implementation_delta.json',dict(status='PASS',
        window_changes=['isolated IO import','fixed branch-specific weight lookup','log actual lambda','log actual unweighted backward scale'],
        optimizer_training_loop_AST_unchanged_except_namespace_and_schema=True,
        warmup_STE_rate_precision_detach_and_loss_formula_unchanged=True,
        source_window_sha256=sha(V22/'window_training.py'),local_window_sha256=sha(ROOT/'window_training.py'),
        source_train_sha256=sha(V22/'train.py'),local_train_sha256=sha(ROOT/'train.py'),
        plan_byte_exact=True,optimizer_updates=0))
    print('IMPLEMENTATION DELTA PASS')
if __name__=='__main__':main()

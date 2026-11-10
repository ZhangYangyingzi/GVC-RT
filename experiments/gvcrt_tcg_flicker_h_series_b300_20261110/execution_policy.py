"""User-authorized execution changes; original numeric experiment settings preserved."""
from common_s2 import *
def h2_training_allowed():
    return load(ROOT/'protocol.json')['H2_gate'].get('training_allowed',False) is True

def verify_checkpoint_protocol(saved_hash):
    current=sha(ROOT/'protocol.json')
    if saved_hash==current:return True
    amendment=load(ROOT/'audits/protocol_revisions/H2_user_directed_start/amendment.json')
    old=Path(amendment['prior_protocol_path'])
    assert sha(old)==amendment['prior_protocol_SHA256']==saved_hash
    before=load(old);now=load(ROOT/'protocol.json')
    assert {k:v for k,v in before.items() if k!='H2_gate'}=={k:v for k,v in now.items() if k!='H2_gate'}
    assert h2_training_allowed() and amendment['numeric_model_loss_data_optimizer_selection_unchanged']
    return True

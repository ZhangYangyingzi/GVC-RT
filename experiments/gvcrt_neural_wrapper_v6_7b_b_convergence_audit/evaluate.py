from v67b_io import *
if __name__=='__main__':
    assert load(ROOT/'audits/fid_protocol_audit.json')['status']=='PASS'
    eval_adapter().main()

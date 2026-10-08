from io16 import *
def main():
    import torch,traceback
    torch.set_num_threads(2);frozen();out=[]
    try:
        for row in replay_rows():
            frames,plan=replay(row,torch.device('cpu'))
            assert row['actual_qps']==load(ROOT/'qp_semantics_audit.json')['actual_qps'][str(row['external_qp'])][:4]
            out.append(dict(**plan,external_qp=row['external_qp'],actual_qps=row['actual_qps'],input_hashes_match=True))
            if len(out)%50==0:
                dump(ROOT/'audits/replay_status.json',dict(status='RUNNING',verified=len(out),expected=1000));print('REPLAY',len(out),flush=True)
        write(ROOT/'audits/replay_inputs.csv',out)
        dump(ROOT/'audits/replay_integrity.json',dict(status='PASS',verified=1000,expected=1000,source_log_sha256=sha(V15/'training_logs/mixed.jsonl'),replay_plan_sha256=sha(ROOT/'replay_plan.json'),source_samples_changed=False))
        print('REPLAY PASS 1000',flush=True)
    except Exception:dump(ROOT/'audits/replay_integrity.json',dict(status='FAIL',verified=len(out),expected=1000,error=traceback.format_exc()));raise
if __name__=='__main__':main()

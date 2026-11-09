"""CPU regression checks for cursor retention, ordered lookahead and detail formula."""
import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import pipeline

class Scheduling(unittest.TestCase):
    def test_original_order_and_rank_drift(self):
        rows=[dict(plan_slot=i,external_qp=i%10,sample_key=str(i)) for i in range(1200)]
        with patch.object(pipeline,'ready',return_value=False):
            got=list(pipeline.pending_jobs(rows,dict(E=36,F=0),dict(prefetch_windows=32)))
        slots=[r['plan_slot'] for r,q,tr in got]
        self.assertEqual(slots[4:],list(range(32)))
        self.assertEqual(set(slots[:4]),set(range(36,40)))
        self.assertEqual(len(set(slots)),36)
    def test_slow_rank_and_recovery_checkpoint_protect_data(self):
        with tempfile.TemporaryDirectory(dir=pipeline.ROOT/'tmp') as tmp:
            root=Path(tmp);(root/'cache/requests').mkdir(parents=True)
            rows=[dict(plan_slot=i,external_qp=0) for i in range(8)]
            for i in range(8):(root/f'{i}.pt').write_bytes(b'fixed_codec')
            def paths(row,q,tr):
                i=row['plan_slot'];return str(i),root/f'{i}.pt',root/f'{i}.json'
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'cache_paths',side_effect=paths):
                pipeline.trim(rows,dict(E=8,F=4))
            self.assertTrue(all(not (root/f'{i}.pt').exists() for i in range(4)))
            self.assertTrue(all((root/f'{i}.pt').exists() for i in range(4,8)))
    def test_detail_is_gt_difference_equal_scale_means(self):
        import torch
        from loss import detail_error
        from temporal import sobel
        torch.manual_seed(42);x=torch.rand(1,3,16,16,requires_grad=True);gt=torch.rand_like(x)
        actual=detail_error(x,gt)
        expected=sum((sobel(torch.nn.functional.avg_pool2d(x,k,k) if k>1 else x)-sobel(torch.nn.functional.avg_pool2d(gt,k,k) if k>1 else gt)).abs().mean() for k in [1,2,4])/3
        self.assertEqual(float(actual.detach()),float(expected.detach()))
        actual.backward();self.assertGreater(float(x.grad.norm()),0)
        self.assertEqual(float(detail_error(gt,gt)),0.)
if __name__=='__main__':unittest.main()

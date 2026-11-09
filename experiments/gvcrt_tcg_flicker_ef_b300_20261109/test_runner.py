"""Publication failure/recovery must never re-enter training or engineering."""
import contextlib,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import runner
from common_s2 import ROOT,dump,load,GROUPS

class PublicationRecovery(unittest.TestCase):
    def test_complete_is_published_only_after_verified_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            root=Path(d);observed=[]
            for g in GROUPS:dump(root/g/'training/completion.json',dict(status='PASS',committed_updates=300))
            dump(root/'publication_remote_receipt.json',dict(status='VERIFIED',commit='verified_results'))
            def publish(phase):
                observed.append((phase,load(root/'status.json')['complete']));return True
            with patch.object(runner,'ROOT',root),patch('summarize.report'),patch('publish.publish',side_effect=publish):runner.finalize_publication()
            self.assertEqual(observed,[('results',False),('results_final',True)])
            self.assertTrue(load(root/'status.json')['complete'])
    def test_failed_results_push_never_marks_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            root=Path(d)
            for g in GROUPS:dump(root/g/'training/completion.json',dict(status='PASS',committed_updates=200))
            with patch.object(runner,'ROOT',root),patch('summarize.report'),patch('publish.publish',return_value=False):
                with self.assertRaises(AssertionError):runner.finalize_publication()
            self.assertFalse(load(root/'status.json')['complete'])
    def test_resume_after_integrity_skips_all_gpu_and_training_stages(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
            root=Path(d);dump(root/'final_integrity.json',dict(status='PASS'))
            with patch.object(runner,'ROOT',root),patch.object(runner,'env'),patch.object(runner,'lock',return_value=contextlib.nullcontext()),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3'),patch('deliver.integrity') as integrity,patch.object(runner,'finalize_publication') as finish,patch.object(runner,'start') as start:
                runner.main()
            integrity.assert_called_once();finish.assert_called_once();start.assert_not_called()
if __name__=='__main__':unittest.main()

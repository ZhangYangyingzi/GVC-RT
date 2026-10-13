# V6.30 execution failure

No updates.

```text
Traceback (most recent call last):
  File "/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/run_all.py", line 168, in main
    integrity=finalize()
              ^^^^^^^^^^
  File "/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/finalize30.py", line 122, in finalize
    subprocess.run(['/data1/anaconda3_new/anaconda_program/bin/python','-B',str(ROOT/'plot30.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
  File "/Huang_group/zyyz/home_dir/.conda/envs/gvc-rt/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/data1/anaconda3_new/anaconda_program/bin/python', '-B', '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_neural_wrapper_v6_30_fid_cache_closure_sample_robustness_audit/plot30.py']' returned non-zero exit status 1.

```

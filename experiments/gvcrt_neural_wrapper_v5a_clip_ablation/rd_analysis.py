from v5_utils import V41,module
source=module('v5_rd_source',V41/'rd_analysis.py')
METRICS=('LPIPS','DISTS','FloLPIPS','FID')
compare=source.compare
gate=source.gate

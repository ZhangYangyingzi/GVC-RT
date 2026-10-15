"""Restore this experiment's compressed text with SHA verification; preserve conflicts."""
import gzip
from audit_io import *
for r in load(ROOT/'publication_artifacts.json')['large_text_gzip']:
 p=ROOT/r['original'];assert p.resolve().is_relative_to(ROOT)
 if p.exists():assert sha(p)==r['sha256'];continue
 assert sha(ROOT/r['gzip'])==r['gzip_sha256'];b=gzip.open(ROOT/r['gzip'],'rb').read();assert hashlib.sha256(b).hexdigest()==r['sha256'];p.write_bytes(b)

"""Restore large local CSV/JSON files compressed for GitHub blob limits."""
import gzip
import shutil
from audit_io import *
if __name__=='__main__':
    for record in load(ROOT/'publication_artifacts.json')['large_text_gzip']:
        source=ROOT/record['compressed'];assert sha(source)==record['compressed_sha256']
        with gzip.open(source,'rb') as src,(ROOT/record['original']).open('wb') as dst:shutil.copyfileobj(src,dst)
        assert sha(ROOT/record['original'])==record['sha256']

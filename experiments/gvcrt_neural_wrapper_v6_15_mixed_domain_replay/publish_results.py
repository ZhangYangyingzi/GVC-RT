"""Complete the existing isolated-worktree publisher, including local gitignore."""
from mixed_io import *
import publish_github

def main():
    assert load(ROOT/'final_integrity.json')['status']=='PASS'
    assert load(ROOT/'audits/completion_checks.json')['status']=='PASS'
    # The original publisher already excludes weights and caches. Include only
    # the intentionally added extensionless .gitignore file as well.
    publish_github.EXCLUDED.update({'decoded_frames','frame_cache'})
    files=[p for p in ROOT.rglob('*') if p.is_file() and not p.suffix
           and not any(x in publish_github.EXCLUDED for x in p.relative_to(ROOT).parts)]
    assert files==[ROOT/'.gitignore'] or set(files)=={ROOT/'.gitignore'},files
    publish_github.ALLOWED.add('')
    imported=load(ROOT/'audits/imported_modules.json')
    virtual=[p for p in imported if not Path(p).is_file()]
    assert all(Path(p).name in ('<stdin>','_classes.py','_ops.py') for p in virtual),virtual
    dump(ROOT/'audits/publication_virtual_modules.json',dict(excluded_virtual_paths=virtual,reason='Python/PyTorch synthetic module __file__ values; no source files exist'))
    original_load=publish_github.load
    def publication_load(path):
        value=original_load(path)
        return [p for p in value if p not in virtual] if Path(path)==ROOT/'audits/imported_modules.json' else value
    publish_github.load=publication_load
    command([PYTHON,'-B','-u',str(Path(__file__).resolve())])
    publish_github.main()

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        dump(ROOT/'publication_status.json',dict(status='FAIL',reason=traceback.format_exc(),unix=time.time()));raise

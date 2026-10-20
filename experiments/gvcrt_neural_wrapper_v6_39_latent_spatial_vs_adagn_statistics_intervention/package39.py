"""Copy only light v6.39 artifacts to an isolated publication worktree."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REL=Path("experiments")/HERE.name
LIMIT=512*1024
SKIP={"tasks","raw","features","artifacts","logs","__pycache__"}
EXCLUDE={"driver.lock","run_status.json"}


def sha(data):return hashlib.sha256(data).hexdigest()


def main(target):
    target=Path(target)/REL
    target.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():continue
        relative=path.relative_to(HERE)
        if any(p in SKIP for p in relative.parts) or path.name in EXCLUDE:continue
        if path.suffix not in (".py",".md",".json",".jsonl",".csv",".png"):continue
        if relative.as_posix() in ("file_index.json",):continue
        raw=path.read_bytes()
        # Plot and fixed-crop PNGs are published as binary figures, never split.
        if path.suffix==".png" and len(raw)>4*1024*1024:continue
        if len(raw)<=LIMIT or path.suffix==".png":
            dest=target/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(raw)
            parts=[relative.as_posix()]
        else:
            assert path.suffix!=".png"
            lines=raw.splitlines(keepends=True)
            assert b"".join(lines)==raw
            chunks=[];current=b""
            for line in lines:
                assert len(line)<=LIMIT
                if len(current)+len(line)>LIMIT:
                    chunks.append(current);current=b""
                current+=line
            if current:chunks.append(current)
            parts=[]
            for i,block in enumerate(chunks):
                relpart=Path("parts")/(relative.as_posix().replace("/","__")+f".part{i:04d}")
                dest=target/relpart;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(block)
                parts.append(relpart.as_posix())
            assert b"".join(chunks)==raw
        files.append(dict(path=relative.as_posix(),bytes=len(raw),sha256=sha(raw),parts=parts,
                          part_bytes=[(target/p).stat().st_size for p in parts],
                          part_sha256=[sha((target/p).read_bytes()) for p in parts]))
    (target/"file_index.json").write_text(json.dumps(dict(limit_bytes=LIMIT,files=files),indent=2)+"\n")
    print("PACKAGE",len(files),"files",target)


if __name__=="__main__":main(sys.argv[1])

"""Reassemble and SHA-verify UTF-8 text split at complete lines."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("published_directory",type=Path)
    parser.add_argument("output_directory",type=Path)
    args=parser.parse_args()
    index=json.loads((args.published_directory/"file_index.json").read_text())
    for item in index["files"]:
        pieces=[args.published_directory/p for p in item["parts"]]
        data=b"".join(p.read_bytes() for p in pieces)
        assert hashlib.sha256(data).hexdigest()==item["sha256"]
        assert len(data)==item["bytes"]
        dest=args.output_directory/item["path"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        print(item["path"],len(data),"PASS")


if __name__=="__main__":main()

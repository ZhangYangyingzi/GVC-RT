"""Fixed source/QP/frame/region previews with measured v3 total rate."""
import csv
import json
from pathlib import Path
from PIL import Image,ImageDraw

P=Path(__file__).resolve().parent


def main():
    with (P/'rate_match_selection.csv').open() as f: choices=list(csv.DictReader(f))
    output=P/'figures/fixed_previews';output.mkdir(parents=True,exist_ok=True)
    for source in ('UVal1','Jockey'):
        for qp in (2,7):
            for frame in (15,47):
                canvas=Image.new('RGB',(5*256,286),'white')
                draw=ImageDraw.Draw(canvas)
                for col,mode in enumerate(('A','MAP','MEAN','ROLL','IDENTITY')):
                    alpha=2
                    if mode in ('MEAN','ROLL'):
                        alpha=int(next(x for x in choices if x['sequence']==source and int(x['external_qp'])==qp and x['mode']==mode)['alpha_index'])
                    key=f'{mode}_s300_{source}_q{qp}_a{alpha}'
                    result=json.loads((P/'server_artifacts/evaluation'/f'{key}.json').read_text())
                    crop=Image.open(P/'previews'/f'{key}_f{frame}.png').convert('RGB')
                    assert crop.size==(256,256)
                    canvas.paste(crop,(col*256,30))
                    draw.text((col*256+4,7),f'{mode} a={result["alpha"]} {result["actual_bpp"]:.5f} bpp',fill='black')
                canvas.save(output/f'{source}_q{qp}_f{frame}_rows200-456_cols400-656.png')


if __name__=='__main__':main()

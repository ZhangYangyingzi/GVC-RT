"""Assemble only the preregistered frames, using saved matched/native RGB previews."""
from common_s2 import *
from evaluate import videos

def main():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    selection=load(ROOT/'selection.json');step=selection['best_trained_joint']['step'];settings=load(ROOT/'protocol.json')['diagnostics'];records=[]
    for split in ['development','final']:
        selected=settings['development_sequences'] if split=='development' else settings['final_sequences']
        for v in videos(split):
            if v['sequence'] not in selected:continue
            key=f'{v["domain"]}_{v["sequence"]}_qp5'
            jd=ROOT/'diagnostics'/evaluation_id(split,step)/key;bd=ROOT/'diagnostics'/evaluation_id(split,candidate='matched_baseline')/key
            for frame in settings['P_frames']:
                paths=[jd/f'{frame:02d}_GT.png',bd/f'{frame:02d}_B1500_matched_FP32.png',jd/f'{frame:02d}_Joint.png']
                images=[Image.open(p).convert('RGB') for p in paths];assert all(im.size==images[0].size for im in images)
                w,h=images[0].size;canvas=Image.new('RGB',(3*w,h+40),'white');draw=ImageDraw.Draw(canvas)
                for i,(im,label) in enumerate(zip(images,['GT','B+1500 matched FP32-TCG',f'Internal history gate joint, selected step{step}'])):
                    canvas.paste(im,(i*w,40));draw.text((i*w+10,10),label,fill='black')
                out=jd/f'{frame:02d}_GT_B1500matched_Joint.png';canvas.save(out)
                records.append(dict(split=split,sequence=v['sequence'],frame=frame,QP=5,comparison=str(out.relative_to(ROOT)),M=str((jd/f'{frame:02d}_M.png').relative_to(ROOT))))
                for p in [*paths,jd/f'{frame:02d}_M.png']:assert p.exists()
    assert len(records)==16
    dump(ROOT/'audits/diagnostics.json',dict(status='PASS',predeclared_only=True,records=records,mask_color_scale=[0,1],
        order='GT | B+1500 matched FP32-TCG | selected trained joint',preview='8-bit RGB lossless PNG container for viewing; never used for neural metrics'))
if __name__=='__main__':env();main()

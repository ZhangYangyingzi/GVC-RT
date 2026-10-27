"""Full-frame overviews for the predeclared frames, with shared center crop marked."""
from PIL import Image, ImageDraw
from common_run import HERE, load


def run():
    for item in load(HERE/'stream_manifest.json'):
        name=f"{item['sequence']}_q{item['external_qp']}"
        for frame in (15,31,63):
            labels=('GT','A','G','R','RQ')
            previews=HERE/'local_artifacts/previews'/name
            images=[]
            for label in labels:
                image=Image.open(previews/f'{label}_{frame:02d}.png').convert('RGB')
                w,h=image.size; x=max(0,(w-256)//2); y=max(0,(h-256)//2)
                ImageDraw.Draw(image).rectangle((x,y,x+255,y+255),outline='red',width=3)
                images.append(image.resize((320,round(h*320/w))))
            height=images[0].height
            board=Image.new('RGB',(1600,height+24),'white');draw=ImageDraw.Draw(board)
            for i,(label,image) in enumerate(zip(labels,images)):
                board.paste(image,(i*320,24));draw.text((i*320+8,6),label,fill='black')
            board.save(HERE/'figures'/f'{name}_overview_frame{frame:02d}.png')
    print('FIXED_OVERVIEWS_PASS')


if __name__ == '__main__':
    run()

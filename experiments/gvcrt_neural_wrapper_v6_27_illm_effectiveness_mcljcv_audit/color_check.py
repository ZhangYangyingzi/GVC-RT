from io21 import *
import numpy as np

def main():
 tests=[]
 for name,rgb in [('gray',[128,128,128]),('red',[255,0,0]),('green',[0,255,0]),('blue',[0,0,255])]:
  r,g,b=np.array(rgb,float)/255;y=.2126*r+.7152*g+.0722*b;u=(b-y)/1.8556;v=(r-y)/1.5748;Y=int(round(16+219*y));U=int(round(128+224*u));V=int(round(128+224*v));raw=bytes([Y])*256+bytes([U])*64+bytes([V])*64
  cmd=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-f','rawvideo','-pixel_format','yuv420p','-video_size','16x16','-i','pipe:0','-vf','scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=limited:out_range=full','-frames:v','1','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];out=subprocess.check_output(cmd,input=raw);a=np.frombuffer(out,dtype=np.uint8).reshape(16,16,3);actual=a.mean((0,1));assert max(abs(actual-np.array(rgb)))<=3,(name,actual);tests.append(dict(test=name,YUV=[Y,U,V],expected_RGB=rgb,actual_RGB=actual.tolist(),max_error=float(max(abs(actual-np.array(rgb)))),status='PASS'))
 dump(ROOT/'audits/color_conversion_integrity.json',dict(status='PASS',matrix='BT709',in_range='limited',out_range='full',channel_order='RGB',synthetic_tests=tests,raw_source_range_embedded_metadata=False,convention='fixed historical pipeline, not inferred intrinsic metadata'))
if __name__=='__main__':main()

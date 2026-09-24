"""ORC2: shared-model/base binding, one-byte zero-grid flags, isolated arithmetic state."""
import struct
import zlib
from support import torch,sha,arithmetic_encode,arithmetic_decode

HEADER=struct.Struct("<4sBBHHHHHHf32s32s")


def checked_symbols(s):
    if s.ndim!=4 or s.shape[0]!=1 or not torch.isfinite(s).all() or not torch.equal(s,s.round()):
        raise ValueError("Expected finite integer NCHW symbols, batch 1")
    if s.min() < -128 or s.max() > 127:
        raise OverflowError(f"No clipping allowed: symbols [{s.min().item()}, {s.max().item()}]")
    return s.detach().cpu().short()


def encode(path,base_path,model_path,method,level,delta,shape,frames,entropy,valid_hw=(1080,1920)):
    # delta is already a fixed float32 calibration value, stored once per sequence.
    if struct.unpack("<f",struct.pack("<f",delta))[0]!=delta or not delta>0:
        raise ValueError("delta must be an exactly represented positive float32")
    c,h,w=shape
    blob=bytearray(HEADER.pack(b"ORC2",method,level,*valid_hw,len(frames),c,h,w,delta,
                               bytes.fromhex(sha(base_path)),bytes.fromhex(sha(model_path))))
    cdfs=None
    stats=[]
    for index,s in enumerate(frames):
        if s is None:
            blob.append(2)  # I frame, unchanged original reconstruction
            stats.append({"frame":index,"kind":"I","actual_bits":8,"payload_bits":0})
            continue
        s=checked_symbols(s)
        if tuple(s.shape)!=(1,c,h,w):
            raise ValueError("Symbol grid mismatch")
        if torch.count_nonzero(s)==0:
            blob.append(0)
            stats.append({"frame":index,"kind":"ZERO","actual_bits":8,"payload_bits":0})
            continue
        if cdfs is None:
            cdfs,_=entropy.coder(delta)
        raw=(s.flatten()+128).tolist()
        payload=arithmetic_encode(raw,cdfs,h*w)
        blob.append(1)
        blob.extend(struct.pack("<II",len(payload),zlib.crc32(payload)))
        blob.extend(payload)
        stats.append({"frame":index,"kind":"CODED","actual_bits":8*(9+len(payload)),"payload_bits":8*len(payload)})
    with open(path,"xb") as f:
        f.write(blob)
    assert len(blob)*8==HEADER.size*8+sum(r["actual_bits"] for r in stats)
    return {"file_bits":len(blob)*8,"header_bits":HEADER.size*8,"frames":stats}


def decode(path,base_path,model_path,entropy):
    with open(path,"rb") as f:
        blob=f.read()
    if len(blob)<HEADER.size:
        raise ValueError("Truncated ORC2")
    magic,method,level,vh,vw,n,c,h,w,delta,basehash,modelhash=HEADER.unpack_from(blob)
    if magic!=b"ORC2" or method not in (0,1) or basehash.hex()!=sha(base_path) or modelhash.hex()!=sha(model_path):
        raise ValueError("Stream/base/shared-model identity mismatch")
    if not (0<n<=256 and 0<c<=64 and 0<h*w<=65536 and 0<delta<float("inf")):
        raise ValueError("Invalid stream dimensions or delta")
    if c!=entropy.log_scale.shape[1]:
        raise ValueError("Entropy channel mismatch")
    pos=HEADER.size
    frames=[]
    cdfs=None
    for _ in range(n):
        if pos>=len(blob):
            raise ValueError("Truncated flag")
        flag=blob[pos]
        pos+=1
        if flag==2:
            frames.append(None)
        elif flag==0:
            frames.append(torch.zeros((1,c,h,w),dtype=torch.int16))
        elif flag==1:
            if pos+8>len(blob):
                raise ValueError("Truncated payload header")
            length,crc=struct.unpack_from("<II",blob,pos)
            pos+=8
            if pos+length>len(blob):
                raise ValueError("Truncated payload")
            payload=blob[pos:pos+length]
            pos+=length
            if zlib.crc32(payload)!=crc:
                raise ValueError("Payload checksum mismatch")
            if cdfs is None:
                cdfs,_=entropy.coder(delta)
            values=arithmetic_decode(payload,cdfs,h*w,c*h*w)
            frames.append((torch.tensor(values,dtype=torch.int16)-128).reshape(1,c,h,w))
        else:
            raise ValueError("Unknown frame flag")
    if pos!=len(blob):
        raise ValueError("Trailing stream bytes")
    return {"method":method,"level":level,"delta":delta,"shape":(c,h,w),"valid_hw":(vh,vw),"frames":frames}

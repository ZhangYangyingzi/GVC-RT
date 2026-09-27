import io
from audit_utils import REPO,module,load
stream=module('v51_stream_helper',REPO/'src/utils/stream_helper.py')

def structure(path,external_qp,count):
    data=path.read_bytes();buffer=io.BytesIO(data);helper=stream.SPSHelper();rows=[]
    audit=load('qp_semantics_audit.json')
    while buffer.tell()<len(data):
        start=buffer.tell();header=stream.read_header(buffer)
        while header['nal_type']==stream.NalType.NAL_SPS:
            helper.add_sps_by_id(stream.read_sps_remaining(buffer,header['sps_id']));header=stream.read_header(buffer)
        sps=helper.get_sps_by_id(header['sps_id']);qp,payload=stream.read_ip_remaining(buffer)
        index=len(rows);is_i=header['nal_type']==stream.NalType.NAL_I
        assert is_i==(index==0) and sps['height']==1088 and sps['width']==1920 and sps['use_ada_i']==0
        expected=external_qp if is_i else external_qp+audit['qp_shift'][audit['index_map'][index%8]]
        assert qp==expected and 0<=qp<12,(index,qp,expected)
        rows.append(dict(frame=index,external_qp=external_qp,actual_i_qp=external_qp,
            actual_p_qp='' if is_i else qp,actual_qp=qp,is_I=is_i,real_bits=(buffer.tell()-start)*8,payload_bits=len(payload)*8))
    assert len(rows)==count and sum(r['real_bits'] for r in rows)==len(data)*8
    return rows

"""Read-only format parser and bit-for-bit instrumented replay of V1 arithmetic coding."""
import json
import math
import struct
from common import HERE,V1,CFG,torch,models,read_container,save_json,write_csv
from evaluation import CONTAINER
from enhancement import arithmetic_encode


def trace_arithmetic(symbols,cdfs,per_channel):
    low,high,pending=0,(1<<32)-1,0
    bits=[]
    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1-bit]*pending)
        pending=0
    for index,symbol in enumerate(symbols):
        cdf=cdfs[index//per_channel]
        span=high-low+1
        high=low+span*cdf[symbol+1]//65536-1
        low+=span*cdf[symbol]//65536
        while True:
            if high<(1<<31):
                emit(0)
            elif low>=(1<<31):
                emit(1)
                low-=1<<31
                high-=1<<31
            elif low>=(1<<30) and high<(3<<30):
                pending+=1
                low-=1<<30
                high-=1<<30
            else:
                break
            low*=2
            high=high*2+1
    # Deferred underflow bits originate in symbol coding; final decision adds 2 bits.
    data_bits=len(bits)+pending
    deferred=pending
    pending+=1
    emit(0 if low<(1<<30) else 1)
    assert len(bits)==data_bits+2
    bits += [0]*32
    alignment=(-len(bits))%8
    bits += [0]*alignment
    payload=bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))
    return payload,{"symbol_coding_bits":data_bits,"deferred_symbol_bits_at_flush":deferred,
                    "termination_decision_bits":2,"lookahead_bits":32,"alignment_bits":alignment,
                    "arithmetic_payload_bits":len(payload)*8}


def run(out):
    _,_,entropy,_=models(CFG["ablation"]["checkpoint"],device="cpu")
    entries=json.loads((HERE/"manifest.json").read_text())["val6"]
    details=[]
    sequences=[]
    probabilities=[]
    total_symbols=0
    total_nonzero=0
    for entry in entries:
        video=entry["video_id"]
        path=V1/"run_v1/quantized_d_val6_l2"/f"{video}.gle"
        base=V1/"run_v1/base"/f"{video}_q0.bin"
        hw,packets=read_container(path,base)
        row={"video":video,"duration_seconds":entry["duration_seconds"],"file_bytes":path.stat().st_size,
             "container_general_bits":(4+4)*8,"container_valid_shape_bits":4*8,"container_base_hash_bits":32*8,
             "container_frame_length_bits":len(packets)*4*8,"frame_general_header_bits":0,
             "frame_shape_quantization_bits":0,"cdf_fingerprint_bits":0,"cdf_table_bits":0,
             "source_mean_scale_bits":0,"symbol_coding_bits":0,"termination_decision_bits":0,
             "lookahead_bits":0,"alignment_bits":0,"arithmetic_payload_bits":0,"estimated_float_bits":0.,
             "ideal_quantized_cdf_bits":0.,"zero_symbols":0,"nonzero_symbols":0,"coder_initializations":0}
        for frame,packet in enumerate(packets):
            if not packet:
                assert frame==0
                continue
            symbols,delta,level=entropy.decode(packet)
            assert delta==2.0 and level==2
            assert tuple(symbols.shape)==(1,8,17,30)
            total_symbols+=symbols.numel()
            nz=int(torch.count_nonzero(symbols))
            total_nonzero+=nz
            row["zero_symbols"]+=symbols.numel()-nz
            row["nonzero_symbols"]+=nz
            cdfs,_=entropy.coder(delta)
            raw=(symbols.flatten()+128).tolist()
            replay,counts=trace_arithmetic(raw,cdfs,17*30)
            payload=packet[entropy.HEADER.size:]
            assert replay==payload and arithmetic_encode(raw,cdfs,17*30)==payload
            estimated=entropy.bits(symbols.float(),delta).item()
            p0=[(cdf[129]-cdf[128])/65536 for cdf in cdfs]
            ideal=sum(-math.log2((cdfs[i//510][s+1]-cdfs[i//510][s])/65536) for i,s in enumerate(raw))
            scales=(torch.nn.functional.softplus(entropy.log_scale.detach().double()).flatten()+1e-4)
            float_p0=(torch.sigmoid(1/scales)-torch.sigmoid(-1/scales)).tolist()
            if not probabilities:
                probabilities=[{"channel":i,"shared_logistic_scale":scales[i].item(),"float_p_zero":float_p0[i],
                                "actual_cdf_p_zero":p0[i],"bits_per_zero_actual_cdf":-math.log2(p0[i]),
                                "scale_transmitted":False} for i in range(8)]
            # GLE1 24-byte header partition: magic/version/reserved/length=10,
            # C/level/H/W/delta=10, fingerprint=4. No CDF tables or means/scales.
            row["frame_general_header_bits"]+=10*8
            row["frame_shape_quantization_bits"]+=10*8
            row["cdf_fingerprint_bits"]+=4*8
            for key in ["symbol_coding_bits","termination_decision_bits","lookahead_bits","alignment_bits","arithmetic_payload_bits"]:
                row[key]+=counts[key]
            row["estimated_float_bits"]+=estimated
            row["ideal_quantized_cdf_bits"]+=ideal
            row["coder_initializations"]+=1
            details.append({"video":video,"frame":frame,"delta":delta,"packet_bytes":len(packet),
                            "header_bits":entropy.HEADER.size*8,"estimated_float_bits":estimated,
                            "ideal_quantized_cdf_bits":ideal,**counts,"replay_matches_file":True})
        header_keys=["container_general_bits","container_valid_shape_bits","container_base_hash_bits",
                     "container_frame_length_bits","frame_general_header_bits","frame_shape_quantization_bits","cdf_fingerprint_bits"]
        row["total_header_bits"]=sum(row[k] for k in header_keys)
        row["actual_bits"]=row["file_bytes"]*8
        assert row["actual_bits"]==row["total_header_bits"]+row["symbol_coding_bits"]+row["termination_decision_bits"]+row["lookahead_bits"]+row["alignment_bits"]
        assert row["arithmetic_payload_bits"]==row["actual_bits"]-row["total_header_bits"]
        row["actual_minus_estimated_bits"]=row["actual_bits"]-row["estimated_float_bits"]
        row["actual_kbps"]=row["actual_bits"]/row["duration_seconds"]/1000
        row["hypothetical_zero_flag_bits"]=row["total_header_bits"]+row["coder_initializations"]*8
        row["hypothetical_saved_bits"]=row["actual_bits"]-row["hypothetical_zero_flag_bits"]
        sequences.append(row)
    assert total_symbols==367200 and total_nonzero==0
    total={"video":"ALL_VAL6"}
    for key in sequences[0]:
        if key not in ["video","actual_kbps"]:
            total[key]=sum(row[key] for row in sequences)
    total["actual_kbps"]=total["actual_bits"]/total["duration_seconds"]/1000
    write_csv(out/"rate_breakdown.csv",sequences+[total])
    write_csv(out/"rate_frames.csv",details)
    save_json(out/"zero_probabilities.json",probabilities)
    save_json(out/"rate_checks.json",{"all_symbols_zero":True,"symbol_count":total_symbols,"all_payload_replays_byte_exact":True,
              "coder_reset":"once per P frame; a single arithmetic state spans all 8 channels; channel-wise fixed CDF, no adaptive learning",
              "fixed_length_or_raw_fallback":False,"source_mean_scale_cdf_tables_transmitted":False,
              "repeated_metadata":"same C/H/W/delta/level/CDF fingerprint and format fields repeat in every P packet",
              "hypothetical_flag":"estimate only, retain all headers + add 1 byte per P frame, omit zero grid arithmetic payload; protocol not modified",
              "hypothetical_zero_flag_total_kbps":total["hypothetical_zero_flag_bits"]/total["duration_seconds"]/1000})
    print("RATE",json.dumps(total),flush=True)

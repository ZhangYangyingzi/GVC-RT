"""ORS1: exact sparse integer recoding with per-frame dense fallback."""
import struct
import time
import zlib

from bridge import torch, sha, ORC2
from enhancement import arithmetic_encode, arithmetic_decode

HEADER = struct.Struct("<4sBBHHHHHHf32s32s")
MAGIC = b"ORS1"
SPARSE_HEADER = struct.Struct("<HII")  # nnz, payload bytes, CRC32
DENSE_HEADER = struct.Struct("<II")    # payload bytes, CRC32


def _put_varuint(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("varuint must be a nonnegative int")
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _get_varuint(payload, pos):
    start = pos
    value = 0
    shift = 0
    for _ in range(5):
        if pos >= len(payload):
            raise ValueError("Truncated canonical varuint")
        byte = payload[pos]
        pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if payload[start:pos] != _put_varuint(value):
                raise ValueError("Noncanonical varuint")
            return value, pos
        shift += 7
    raise ValueError("Varuint too long")


def _zigzag(value):
    if not -128 <= value <= 127 or value == 0:
        raise ValueError("Sparse values must be nonzero signed int8")
    return 2 * value if value >= 0 else -2 * value - 1


def _unzigzag(value):
    result = value // 2 if value % 2 == 0 else -(value // 2) - 1
    if not -128 <= result <= 127 or result == 0:
        raise ValueError("Decoded sparse value outside nonzero int8")
    return result


def sparse_payload(symbols):
    flat = ORC2.checked_symbols(symbols).flatten()
    indices = torch.nonzero(flat, as_tuple=False).flatten().tolist()
    if not indices:
        return b"", 0
    payload = bytearray()
    previous = -1
    for index in indices:
        payload.extend(_put_varuint(index - previous))
        payload.extend(_put_varuint(_zigzag(int(flat[index]))))
        previous = index
    return bytes(payload), len(indices)


def decode_sparse_payload(payload, nnz, count, shape):
    if not 0 < nnz <= count:
        raise ValueError("Invalid sparse nonzero count")
    flat = torch.zeros(count, dtype=torch.int16)
    pos = 0
    previous = -1
    for _ in range(nnz):
        gap, pos = _get_varuint(payload, pos)
        if gap <= 0:
            raise ValueError("Sparse positions must increase")
        index = previous + gap
        if index >= count:
            raise ValueError("Sparse position out of range")
        encoded, pos = _get_varuint(payload, pos)
        flat[index] = _unzigzag(encoded)
        previous = index
    if pos != len(payload):
        raise ValueError("Trailing sparse payload bytes")
    return flat.reshape((1,) + tuple(shape))


def packet_candidates(symbols, entropy, delta):
    symbols = ORC2.checked_symbols(symbols)
    if torch.count_nonzero(symbols) == 0:
        return {"zero": b"\x00"}
    c, h, w = symbols.shape[1:]
    cdfs, _ = entropy.coder(delta)
    dense_payload = arithmetic_encode((symbols.flatten() + 128).tolist(), cdfs, h * w)
    dense = b"\x01" + DENSE_HEADER.pack(len(dense_payload), zlib.crc32(dense_payload)) + dense_payload
    payload, nnz = sparse_payload(symbols)
    sparse = b"\x03" + SPARSE_HEADER.pack(nnz, len(payload), zlib.crc32(payload)) + payload
    return {"dense": dense, "sparse": sparse}


def encode(path, base_path, model_path, level, delta, shape, frames, entropy, valid_hw=(1080, 1920)):
    if struct.unpack("<f", struct.pack("<f", delta))[0] != delta or not delta > 0:
        raise ValueError("delta must be an exactly represented positive float32")
    c, h, w = shape
    blob = bytearray(HEADER.pack(MAGIC, 1, level, *valid_hw, len(frames), c, h, w, delta,
                                 bytes.fromhex(sha(base_path)), bytes.fromhex(sha(model_path))))
    stats = []
    encode_start = time.perf_counter()
    for frame, symbols in enumerate(frames):
        if symbols is None:
            blob.append(2)
            stats.append({"frame": frame, "kind": "I", "actual_bits": 8, "payload_bits": 0})
            continue
        symbols = ORC2.checked_symbols(symbols)
        if tuple(symbols.shape) != (1, c, h, w):
            raise ValueError("Symbol grid mismatch")
        candidates = packet_candidates(symbols, entropy, delta)
        if "zero" in candidates:
            mode, packet = "ZERO", candidates["zero"]
        elif len(candidates["sparse"]) < len(candidates["dense"]):
            mode, packet = "SPARSE", candidates["sparse"]
        else:
            mode, packet = "DENSE", candidates["dense"]
        blob.extend(packet)
        header_bytes = 1 if mode == "ZERO" else (11 if mode == "SPARSE" else 9)
        stats.append({"frame": frame, "kind": mode, "actual_bits": 8 * len(packet),
                      "packet_header_bits": 8 * header_bytes,
                      "payload_bits": 8 * (len(packet) - header_bytes),
                      "dense_candidate_bits": 8 * len(candidates.get("dense", packet)),
                      "sparse_candidate_bits": 8 * len(candidates.get("sparse", packet))})
    with open(path, "xb") as f:
        f.write(blob)
    return {"file_bits": len(blob) * 8, "header_bits": HEADER.size * 8, "frames": stats,
            "encode_and_mode_select_seconds": time.perf_counter() - encode_start}


def decode(path, base_path, model_path, entropy):
    start = time.perf_counter()
    blob = open(path, "rb").read()
    if len(blob) < HEADER.size:
        raise ValueError("Truncated ORS1")
    magic, version, level, vh, vw, n, c, h, w, delta, basehash, modelhash = HEADER.unpack_from(blob)
    if magic != MAGIC or version != 1 or basehash.hex() != sha(base_path) or modelhash.hex() != sha(model_path):
        raise ValueError("ORS1 stream/base/shared-model identity mismatch")
    if not (0 < n <= 256 and 0 < c <= 64 and 0 < h * w <= 65536 and 0 < delta < float("inf")):
        raise ValueError("Invalid ORS1 dimensions or delta")
    if c != entropy.log_scale.shape[1]:
        raise ValueError("Entropy channel mismatch")
    pos = HEADER.size
    frames = []
    cdfs = None
    for _ in range(n):
        if pos >= len(blob):
            raise ValueError("Truncated frame mode")
        mode = blob[pos]
        pos += 1
        if mode == 2:
            frames.append(None)
        elif mode == 0:
            frames.append(torch.zeros((1, c, h, w), dtype=torch.int16))
        elif mode == 1:
            if pos + DENSE_HEADER.size > len(blob):
                raise ValueError("Truncated dense header")
            length, crc = DENSE_HEADER.unpack_from(blob, pos)
            pos += DENSE_HEADER.size
            if pos + length > len(blob):
                raise ValueError("Truncated dense payload")
            payload = blob[pos:pos + length]
            pos += length
            if zlib.crc32(payload) != crc:
                raise ValueError("Dense CRC mismatch")
            if cdfs is None:
                cdfs, _ = entropy.coder(delta)
            values = arithmetic_decode(payload, cdfs, h * w, c * h * w)
            frames.append((torch.tensor(values, dtype=torch.int16) - 128).reshape(1, c, h, w))
        elif mode == 3:
            if pos + SPARSE_HEADER.size > len(blob):
                raise ValueError("Truncated sparse header")
            nnz, length, crc = SPARSE_HEADER.unpack_from(blob, pos)
            pos += SPARSE_HEADER.size
            if pos + length > len(blob):
                raise ValueError("Truncated sparse payload")
            payload = blob[pos:pos + length]
            pos += length
            if zlib.crc32(payload) != crc:
                raise ValueError("Sparse CRC mismatch")
            frames.append(decode_sparse_payload(payload, nnz, c * h * w, (c, h, w)))
        else:
            raise ValueError("Unknown ORS1 frame mode")
    if pos != len(blob):
        raise ValueError("Trailing ORS1 bytes")
    return {"level": level, "delta": delta, "shape": (c, h, w), "valid_hw": (vh, vw),
            "frames": frames, "parse_seconds": time.perf_counter() - start}

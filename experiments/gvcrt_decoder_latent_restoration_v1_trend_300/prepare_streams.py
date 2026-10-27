"""Verify and reuse the eight v5 A streams; remove only its experiment header."""
from pathlib import Path

from common_run import HERE, dump, load, sha

V5 = Path('/Huang_group/zyyz/home_dir/tmp/gvcrt-quant-v5-objective/experiments/gvcrt_generation_aware_quant_v5_objective_matched_trend_300')


def run():
    audit = load(V5 / 'A_reuse_audit.json')
    assert audit['status'] == 'PASS' and len(audit['rows']) == 8
    sources = {x['id']: x for x in load(HERE / 'evaluation_sources.json')}
    rows = []
    for source in audit['rows']:
        assert source['status'] == 'PASS' and source['frames'] == 64
        assert source['sequence'] in sources and source['external_qp'] in (2, 7)
        assert source['height'] == sources[source['sequence']]['height']
        assert source['width'] == sources[source['sequence']]['width']
        src = sources[source['sequence']]
        original_receipt = load(source['v4_receipt'])
        assert original_receipt['status'] == 'PASS' and original_receipt['mode'] == 'A'
        assert original_receipt['checkpoint_sha256'] == load(HERE / 'weight_manifest.json')[
            '/Huang_group/zyyz/Projects/GVC-RT/experiments/gvcrt_generation_aware_quant_visual_condition_v1_trend_300/server_artifacts/A/step_300.pt']['sha256']
        assert original_receipt['source_sha256'] == src['source_sha256']
        assert original_receipt['source_frame_indices'] == src['indices']
        assert original_receipt['stream_sha256'] == source['native_stream_sha256']
        assert original_receipt['real_total_bytes'] == source['real_total_bytes'] - 16
        if src['file_type'] == 'mp4':
            assert sha(src['path']) == src['source_sha256']
        else:
            for position, index in enumerate(src['indices']):
                name = f'im{position+1:05d}.png' if src['id'] == 'ParkScene' else f'frame_{position:06d}.png'
                assert sha(Path(src['path']) / name) == (src.get('frame_png_sha256') or src['frame_file_sha256'])[position]
        old = Path(source['stream_path'])
        assert sha(old) == source['stream_sha256']
        raw = old.read_bytes()
        assert raw[:6] == b'GV5Q\x01\x00' and source['experiment_header_bytes'] == 16
        native = raw[16:]
        out = HERE / 'local_artifacts/evaluation_streams' / f"{source['sequence']}_q{source['external_qp']}.bin"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(native)
        assert sha(out) == source['native_stream_sha256']
        rows.append(dict(sequence=source['sequence'], external_qp=source['external_qp'],
                         source_path=str(old), source_stream_sha256=source['stream_sha256'],
                         stream_path=str(out), stream_sha256=sha(out),
                         total_native_bytes=out.stat().st_size,
                         bpp=out.stat().st_size * 8 / (64 * source['height'] * source['width']),
                         source_video_sha256=sources[source['sequence']]['source_sha256'],
                         provenance_audit=str(V5 / 'A_reuse_audit.json')))
    dump(HERE / 'stream_manifest.json', rows)
    dump(HERE / 'public_decode_config.json',
         {key: dict(height=value['height'], width=value['width'], frames=64)
          for key, value in sources.items()})
    print('STREAMS_VERIFIED', len(rows), flush=True)


if __name__ == '__main__':
    run()

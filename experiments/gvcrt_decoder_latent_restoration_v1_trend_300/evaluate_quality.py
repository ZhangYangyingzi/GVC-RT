"""Read GT only after standalone decoder receipts and RGB files exist."""
import argparse
import math

from common_run import HERE, dump, load, setup, sha, table


def frames_for(source, torch, np):
    import cv2
    from PIL import Image
    output = []
    if source['file_type'] == 'mp4':
        assert sha(source['path']) == source['source_sha256']
        cap = cv2.VideoCapture(source['path'])
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, source['indices'][0])
        try:
            for index in source['indices']:
                ok, bgr = cap.read()
                assert ok, (source['id'], index)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                output.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255)
        finally:
            cap.release()
    else:
        for position, index in enumerate(source['indices']):
            name = f'im{position+1:05d}.png' if source['id'] == 'ParkScene' else f'frame_{position:06d}.png'
            path = __import__('pathlib').Path(source['path']) / name
            expected = source.get('frame_png_sha256') or source.get('frame_file_sha256')
            if expected is not None:
                assert sha(path) == expected[position]
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
            output.append(torch.from_numpy(rgb).permute(2, 0, 1)[None].float()/255)
    assert len(output) == 64
    assert all(x.shape == (1, 3, source['height'], source['width']) for x in output)
    return output


def run(source_id, qp, branch, gpu):
    torch, np = setup(gpu)
    import cv2
    from PIL import Image
    from core import quality_models, unit
    import flolpips_compat
    source = next(x for x in load(HERE / 'evaluation_sources.json') if x['id'] == source_id)
    path = HERE / 'local_artifacts/decoded' / f'{source_id}_q{qp}' / branch
    receipt = load(path / 'receipt.json')
    assert receipt['branch'] == branch and receipt['sequence'] == source_id
    assert receipt['frames'] == 64 and len(list(path.glob('[0-9][0-9].pt'))) == 64
    assert sha(receipt['stream_path']) == receipt['stream_sha256']
    frames = frames_for(source, torch, np)
    quality = quality_models(torch.device('cuda:0'))
    flow, perceptual = flolpips_compat.load_models(torch.device('cuda:0'))
    result_dir = HERE / 'local_artifacts/quality' / f'{source_id}_q{qp}' / branch
    result_dir.mkdir(parents=True, exist_ok=True)
    movie = cv2.VideoWriter(str(result_dir / 'display.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                            25, (source['width'], source['height']))
    assert movie.isOpened()
    records = []; transitions = []
    prior_gt = prior_output = None
    with torch.inference_mode():
        for i, gt_cpu in enumerate(frames):
            raw = torch.load(path / f'{i:02d}.pt', map_location='cpu', weights_only=True).cuda()
            output = unit(raw, source['height'], source['width']).float()
            gt = gt_cpu.cuda()
            assert output.shape == gt.shape
            error = output - gt
            sse = float(error.square().sum())
            mse = sse / output.numel()
            record = dict(branch=branch, sequence=source_id, external_qp=qp, frame=i,
                          source_frame_index=source['indices'][i], actual_qp=receipt['states'][i]['actual_qp'],
                          pixel_SSE=sse, MSE=mse, PSNR=-10 * math.log10(max(mse, 1e-15)),
                          LPIPS=float(quality[0](output, gt, normalize=True).mean()),
                          DISTS=float(quality[1](output, gt).mean()))
            records.append(record)
            if i:
                motion = flow(prior_gt, gt) - flow(prior_output, output)
                value = float(perceptual(prior_gt, prior_output, motion, normalize=True).item())
                assert math.isfinite(value)
                transitions.append(dict(branch=branch, sequence=source_id, external_qp=qp,
                                        from_frame=i-1, to_frame=i, FloLPIPS=value))
            prior_gt, prior_output = gt, output
            rgb = (output[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype('uint8')
            movie.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if i in (15, 31, 63):
                preview = HERE / 'local_artifacts/previews' / f'{source_id}_q{qp}'
                preview.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(preview / f'{branch}_{i:02d}.png')
                if branch == 'A':
                    grgb = (gt[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype('uint8')
                    Image.fromarray(grgb).save(preview / f'GT_{i:02d}.png')
    movie.release()
    assert len(records) == 64 and len(transitions) == 63
    table(result_dir / 'frames.csv', records)
    table(result_dir / 'transitions.csv', transitions)

    def aggregate(selected):
        return dict(PSNR=-10*math.log10(max(sum(x['pixel_SSE'] for x in selected) /
                                          (len(selected)*source['height']*source['width']*3), 1e-15)),
                    LPIPS=sum(x['LPIPS'] for x in selected)/len(selected),
                    DISTS=sum(x['DISTS'] for x in selected)/len(selected),
                    FloLPIPS=sum(x['FloLPIPS'] for x in transitions)/63)
    summary = dict(branch=branch, sequence=source_id, external_qp=qp,
                   stream_sha256=receipt['stream_sha256'], real_total_bytes=receipt['real_total_bytes'],
                   bpp=receipt['bpp'], all64=aggregate(records), P63=aggregate(records[1:]),
                   frame_csv=str(result_dir / 'frames.csv'), transition_csv=str(result_dir / 'transitions.csv'),
                   display_video=str(result_dir / 'display.mp4'), decoded_receipt=str(path / 'receipt.json'))
    dump(result_dir / 'summary.json', summary)
    print('QUALITY', source_id, qp, branch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--qp', type=int, required=True)
    parser.add_argument('--branch', choices=['A', 'G', 'R', 'RQ'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(args.source, args.qp, args.branch, args.gpu)

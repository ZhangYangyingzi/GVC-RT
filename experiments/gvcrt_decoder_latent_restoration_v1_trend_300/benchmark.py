"""Short same-device output-branch timing, excluding initialization and writes."""
import argparse
import statistics
import time

from common_run import HERE, dump, load_base, setup
from restoration import Adapter, condition


def run(gpu):
    torch, _ = setup(gpu)
    _, pm = load_base(torch)
    record = torch.load(HERE / 'local_artifacts/training_cache/0000.pt',
                        map_location='cpu', weights_only=False)['records'][0]
    f, u, qa, qr = (record[k].cuda() for k in ('f', 'u', 'q_actual', 'q_recon'))
    generator = pm.recon_generation_net.decoder
    output = {}
    with torch.inference_mode():
        for branch in ('A', 'R', 'RQ'):
            adapter = None
            if branch != 'A':
                state = torch.load(HERE / 'local_artifacts/checkpoints' / branch / 'step300.pt',
                                   map_location='cpu', weights_only=False)
                adapter = Adapter().cuda().eval()
                adapter.load_state_dict(state['model'], strict=True)
                d = condition(qa, branch)

            def step():
                x = u if adapter is None else u + adapter(f, u, d)
                return generator(x.float(), qr.float())

            for _ in range(5):
                step()
            torch.cuda.synchronize()
            timings = []
            for _ in range(20):
                torch.cuda.synchronize(); begin = time.perf_counter()
                step(); torch.cuda.synchronize()
                timings.append((time.perf_counter() - begin)*1000)
            output[branch] = dict(median_ms=statistics.median(timings), samples_ms=timings)
    output['adapter_extra_ms_R'] = output['R']['median_ms']-output['A']['median_ms']
    output['adapter_extra_ms_RQ'] = output['RQ']['median_ms']-output['A']['median_ms']
    output['physical_gpu'] = gpu
    output['exclusions'] = 'initialization, first five warmup forwards, metrics, file I/O'
    dump(HERE / 'output_branch_timing.json', output)
    print('TIMING_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    run(parser.parse_args().gpu)

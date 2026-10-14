"""Read-only hourly supervision; writes only monitoring reports, never experiment state.

There is no chat/email notification transport configured. Reports are local artifacts.
The evaluation driver and its checkpoints/code are not modified by this monitor.
"""
import argparse
import fcntl
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'monitoring'
TZ = ZoneInfo('Asia/Shanghai')
TERMINAL = {'COMPLETE', 'COMPLETE_LOCAL_PUSH_FAILED', 'FAILED'}


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def dump(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def process(pid, expected):
    if not pid:
        return dict(pid=None, alive=False)
    try:
        os.kill(pid, 0)
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return dict(pid=pid, alive=state != 'Z' and expected in command, process_state=state)
    except (OSError, ValueError):
        return dict(pid=pid, alive=False)


def check():
    now = datetime.now(TZ)
    status = load(ROOT/'status.json', {})
    launch = load(ROOT/'launch_check.json', {})
    previous = load(OUT/'latest.json', {})
    counts = Counter()
    latest_receipt = None
    for p in (ROOT/'parts').rglob('point.json'):
        r = load(p, {})
        if r.get('independent_decode_pass') and r.get('real_RANS') and r.get('latent_rows') == 64:
            counts[str(r['task']['gpu'])] += 1
            stamp = p.stat().st_mtime
            if latest_receipt is None or stamp > latest_receipt['mtime']:
                latest_receipt = dict(task=r['task'], mtime=stamp, finished=datetime.fromtimestamp(stamp, TZ).isoformat())
    tasks = load(ROOT/'task_manifest.json', {}).get('tasks', [])
    totals = Counter(str(t['gpu']) for t in tasks)
    driver = process(status.get('background_pid', launch.get('driver_pid')), 'run_all.py')
    workers = []
    notes = []
    phase = status.get('status', 'UNKNOWN')
    for w in launch.get('workers', status.get('gpu_workers', [])):
        gpu = w['gpu']
        report = process(w['pid'], f'--gpu {gpu}')
        progress = load(ROOT/'worker_status'/f'gpu{gpu}.json', {})
        log = ROOT/'logs'/f'gpu{gpu}.log'
        report.update(gpu=gpu, completed=counts[str(gpu)], expected=totals[str(gpu)],
                      worker_status=progress.get('status', 'FIRST_POINT_OR_STARTUP'),
                      log_bytes=log.stat().st_size if log.exists() else 0,
                      log_updated=datetime.fromtimestamp(log.stat().st_mtime, TZ).isoformat() if log.exists() else None)
        workers.append(report)
        if phase in ('RUNNING', 'WORKER_FAILURE') and not report['alive'] and progress.get('status') != 'PASS':
            notes.append(f'GPU{gpu} worker 不存活，且没有完成记录')
    completed = sum(counts.values())
    if phase not in TERMINAL and not driver['alive']:
        notes.append('主进程不存活，但实验尚无终态记录')
    if phase == 'WORKER_FAILURE':
        notes.append('主流程报告 worker 失败；检查 status.json 和对应 GPU 日志')
    if phase == 'RUNNING' and previous and completed == previous.get('completed_points'):
        notes.append('本次巡检间隔内没有新增完整 point receipt；可能仍在单点计算，需要结合进程与日志判断')
    publication = load(ROOT/'publication_status.json', {})
    if phase == 'COMPLETE_LOCAL_PUSH_FAILED':
        notes.append('评估已完成，但 GitHub push 未成功，本地结果已保留')
    if phase == 'FAILED':
        notes.append('实验主流程失败；错误详情在 status.json')
    result = dict(checked_at=now.isoformat(), timezone='Asia/Shanghai', phase=phase,
                  completed_points=completed, expected_points=len(tasks),
                  progress_percent=round(completed/max(1,len(tasks))*100, 2), driver=driver,
                  workers=workers, latest_receipt=latest_receipt,
                  sanity=load(ROOT/'sanity/proxy_sanity.json', {}).get('status'),
                  integrity=load(ROOT/'integrity.json', {}).get('status'),
                  publication=publication.get('status', 'NOT_ATTEMPTED'), notes=notes,
                  notification_transport='local_reports_only; no outbound chat/email configured',
                  next_check=(now+timedelta(hours=1)).isoformat() if phase not in TERMINAL else None)
    filename=now.strftime('%Y%m%d_%H%M%S')
    dump(OUT/'history'/f'{filename}.json', result)
    dump(OUT/'latest.json', result)
    lines=[f'巡检时间：{now:%Y-%m-%d %H:%M:%S}（北京时间）', f'实验状态：{phase}',
           f'完整编码点：{completed}/{len(tasks)}（{result["progress_percent"]}%）',
           f'Sanity：{result["sanity"]}；最终完整性：{result["integrity"]}；发布：{result["publication"]}', '']
    for w in workers:
        lines.append(f'- GPU{w["gpu"]}：{w["completed"]}/{w["expected"]}；PID {w["pid"]}；存活={w["alive"]}')
    lines += ['', '异常/提醒：'+('；'.join(notes) if notes else '本次未发现进程退出或已记录的主流程故障。'),
              '下一次巡检：'+str(result['next_check']), '通知方式：本地报告；未配置主动聊天/邮件通知。', '']
    text='\n'.join(lines)
    (OUT/'latest.md').write_text(text)
    print(text, flush=True)
    return phase in TERMINAL


def main():
    p=argparse.ArgumentParser();p.add_argument('--once', action='store_true');args=p.parse_args()
    (OUT/'history').mkdir(exist_ok=True)
    lock=(OUT/'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(OUT/'monitor_status.json', dict(status='RUNNING', pid=os.getpid(), interval_seconds=3600,
         started=datetime.now(TZ).isoformat(), notification_transport='local_reports_only'))
    while True:
        if check() or args.once:
            dump(OUT/'monitor_status.json', dict(status='FINISHED', pid=os.getpid(), finished=datetime.now(TZ).isoformat()))
            return
        deadline=time.monotonic()+3600
        while time.monotonic()<deadline:
            time.sleep(max(0, min(30, deadline-time.monotonic())))


if __name__ == '__main__':
    main()

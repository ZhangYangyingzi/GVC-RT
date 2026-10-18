"""Small progress inspection without importing torch or reading large caches."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
def read(p):
    try:return json.loads(p.read_text())
    except Exception:return {}
state=read(ROOT/'driver_state.json')
print(json.dumps({'status':state.get('status','NOT_STARTED'),'completed':len(state.get('completed',[])),'expected':state.get('expected',90),'pending':state.get('pending'),'active':state.get('active'),'updated':state.get('updated'),'sanity':read(ROOT/'audits/sanity.json').get('status'),'failed':state.get('failed',{})},indent=2))

from common_s2 import *
import argparse,traceback

if __name__=='__main__':
    env();p=argparse.ArgumentParser();p.add_argument('request');a=p.parse_args()
    try:
        from cache_codec import ensure
        request=load(a.request)
        for item in request:
            ensure(item['row'],item['qp'],item['training'],0)
    except Exception as e:
        dump(ROOT/'audits/cache_failures'/f'{os.getpid()}.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),request=a.request,time=time.time()))
        raise

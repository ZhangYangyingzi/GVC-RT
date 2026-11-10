from common_s2 import *
if __name__=='__main__':
    env();job=load(sys.argv[1]);os.environ['MODEL_KIND']='original'
    from cache_codec import ensure
    ensure(job['row'],job['qp'],job['training'],0)
    if job['training']:
        from cache_teacher import ensure as teacher_ensure
        teacher_ensure(job['row'],job['qp'],0)

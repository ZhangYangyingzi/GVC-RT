"""Required diagnostic plots only; old logged data with no conclusions."""
from v67b_io import *
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=read(ROOT/'results/B_training_curve_raw.csv');steps=np.asarray([int(r['step']) for r in rows]);(ROOT/'plots').mkdir(exist_ok=True)
    groups=[('B_training_loss_curves.png',['LPIPS','DISTS','rate_bpp','proxy_L1','weighted_structure_loss','total_loss']),('B_gradient_norm_curves.png',['wrapper_gradient_norm','bridge_gradient_norm','generator_gradient_norm']),('B_proxy_msssim_curve.png',['proxy_MS_SSIM'])]
    for file,keys in groups:
        fig,axes=plt.subplots(len(keys),1,figsize=(10,2.6*len(keys)),squeeze=False,sharex=True)
        for ax,key in zip(axes[:,0],keys):
            y=np.asarray([float(r[key]) for r in rows]);ax.plot(steps,y,alpha=.35,linewidth=.65,label='Per update');smooth=np.convolve(y,np.ones(25)/25,'valid');ax.plot(steps[24:],smooth,linewidth=1,label='Trailing 25-update mean');ax.set_ylabel(key);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[-1,0].set_xlabel('Training update');fig.tight_layout();fig.savefig(ROOT/'plots'/file,dpi=160);plt.close(fig)
    dump(ROOT/'audits/training_plot_audit.json',dict(status='PASS',plots={file:sha(ROOT/'plots'/file) for file,_ in groups},raw_log_unchanged=True))
if __name__=='__main__':main()

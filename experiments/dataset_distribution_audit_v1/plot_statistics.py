"""Measured distributions and training-fitted projections; no interpretation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_io import ROOT,load,read,dump,sha
LABELS={'ulong_train1024':'U-Long train1024','ulong_unused1024':'U-Long unused1024','vimeo_train2048':'Vimeo train2048',
        'uvg_reference_test':'UVG reference','virat_reference':'VIRAT unique reference','virat720':'VIRAT720 codec','virat480':'VIRAT480 codec'}
COLORS=['#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7','#AA4499','#999933']
def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    source=read(ROOT/'complete_source_statistics.csv');codec=read(ROOT/'codec_difficulty_features.csv');files=[]
    def ecdf(rows,feature,path,title):
        fig,ax=plt.subplots(figsize=(9,6))
        for index,dataset in enumerate(sorted({r['dataset'] for r in rows})):
            values=sorted(float(r[feature]) for r in rows if r['dataset']==dataset and r.get(feature,'') not in ('','None'))
            if not values:continue
            assert np.isfinite(values).all()
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{LABELS[dataset]} (N={len(values)})',color=COLORS[index%len(COLORS)])
        ax.set_xlabel(feature);ax.set_ylabel('Empirical cumulative probability');ax.set_ylim(0,1.02)
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        p=ROOT/'plots'/path;p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=180);plt.close(fig);files.append(p)
    specs=[('temporal_RGB_L1','temporal_l1_distribution.png'),('temporal_RGB_MSE','temporal_mse_distribution.png'),
        ('flow_magnitude_mean','flow_magnitude_distribution.png'),('global_translation_normalized','global_motion_distribution.png'),
        ('compensated_residual_L1','compensated_residual_distribution.png'),('sobel_mean','spatial_complexity_distribution.png')]
    for view in ('standardized_content_view','actual_codec_input_view'):
        rows=[r for r in source if r['view']==view]
        for feature,name in specs:
            path=name if view=='standardized_content_view' else str(Path('actual_codec_input_view')/name)
            ecdf(rows,feature,path,view.replace('_',' '))
    for q in (0,4,9):
        ecdf(codec,f'bpp_q{q}',f'codec_bpp_q{q}_distribution.png',f'Original GVC-RT | QP{q} | Vimeo7 / U-Long32 / frozen reference lengths')
        ecdf(codec,f'DISTS_q{q}',f'original_dists_q{q}_distribution.png',f'Original DISTS | QP{q} | measured reconstruction difficulty proxy')
    coords=read(ROOT/'pca_coordinates.csv');var=read(ROOT/'pca_explained_variance.csv');fig,ax=plt.subplots(figsize=(10,7))
    for i,d in enumerate(sorted({r['dataset'] for r in coords})):
        own=[r for r in coords if r['dataset']==d];ref='reference' in d
        ax.scatter([float(r['PC1']) for r in own],[float(r['PC2']) for r in own],label=f'{LABELS[d]} (N={len(own)})',
            color=COLORS[i],s=75 if ref else 18,alpha=1 if ref else .45,marker='X' if ref else 'o')
        if ref:
            for r in own:ax.annotate(r['sample_id'].replace('uvg_',''),(float(r['PC1']),float(r['PC2'])),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.set_xlabel(f'PC1 ({float(var[0]["explained_variance_ratio"]):.1%})');ax.set_ylabel(f'PC2 ({float(var[1]["explained_variance_ratio"]):.1%})')
    ax.set_title('PCA fit only on complete candidate-training features');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    p=ROOT/'plots/pca_train_and_reference.png';fig.savefig(p,dpi=180);plt.close(fig);files.append(p)
    wide=read(ROOT/'complete_feature_cohort.csv');uvg=[r for r in wide if r['dataset']=='uvg_reference_test'];features=load(ROOT/'feature_scaler.json')['features']
    matrix=np.asarray([[float(r['z_'+f]) for f in features] for r in uvg]);limit=float(np.abs(matrix).max()) or 1
    fig,ax=plt.subplots(figsize=(15,5));im=ax.imshow(matrix,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(features)),features,rotation=60,ha='right',fontsize=8);ax.set_yticks(range(len(uvg)),[r['sample_id'].replace('uvg_','') for r in uvg])
    ax.set_title('UVG features standardized by candidate-training mean/std (no clipping)');fig.colorbar(im,ax=ax,label='Training-standardized value');fig.tight_layout()
    p=ROOT/'plots/uvg_per_sequence_feature_radar/standardized_feature_heatmap.png';p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=180);plt.close(fig);files.append(p)
    dump(ROOT/'plot_audit.json',dict(status='PASS',files={str(p.relative_to(ROOT)):sha(p) for p in files},automatic_measured_axis_ranges=True,
        histogram_type='ECDF includes every finite value',reference_not_used_to_fit_PCA_or_scaler=True))
    print('PLOTS PASS',len(files),flush=True)
if __name__=='__main__':main()

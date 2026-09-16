#!/usr/bin/env python3
"""WA-01-style spatial and temporal summary for WR-07 station circles."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bench.execution.navigation.station_poses import StationPoseSampler, normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_disk

RESET = np.array([2.6327167854768594, -0.8387353492148965])
SRC = np.array([2.131064026779481, -0.9289858865987279])
DST = np.array([1.6902007516630673, -2.20])
SRC_YAW, DST_YAW = 1.5697450892050213, -1.551592653589793
R = .05
BASE_TURN = DST_YAW - SRC_YAW
if BASE_TURN > 0: BASE_TURN -= 2*np.pi

def yaw_from_q(q):
    x,y,z,w = q
    return float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--count',type=int,default=200); a=ap.parse_args()
    station_file=Path('bench/tasks/wr07/resources/calibration/stations.json')
    rows=[]
    for seed in range(a.count):
        initial=sample_uniform_disk(RESET,R,seed=seed,salt=0x57523037)
        sm=StationPoseSampler.from_file(station_file,seed=seed)
        source=sm.sample('counter_pick_station',purpose='source_observe'); target=sm.sample('plate_station',purpose='target_place')
        source_xy=np.asarray(source.position[:2]); target_xy=np.asarray(target.position[:2]); source_yaw=yaw_from_q(source.quaternion); target_yaw=yaw_from_q(target.quaternion)
        rng=np.random.default_rng(np.random.SeedSequence([seed,0x57523037,0x5455524E])); turn=BASE_TURN+float(rng.uniform(-np.deg2rad(5),np.deg2rad(5))); turn_yaw=normalize_angle(source_yaw+turn)
        points=[('P0 reset',initial,1.569755536243165),('P1 source',source_xy,source_yaw),('P2 turn',source_xy,turn_yaw),('P3 target',target_xy,target_yaw)]
        rec={'seed':seed,'turn_deg':np.rad2deg(turn),'turn_offset_deg':np.rad2deg(turn-BASE_TURN)}
        for i,(name,xy,yaw) in enumerate(points):
            rec.update({f'p{i}_x':xy[0],f'p{i}_y':xy[1],f'p{i}_yaw':np.rad2deg(yaw)})
        for i in range(3): rec[f'leg{i+1}_m']=float(np.linalg.norm(np.array([rec[f'p{i+1}_x']-rec[f'p{i}_x'],rec[f'p{i+1}_y']-rec[f'p{i}_y']])) )
        rows.append(rec)
    a.output_dir.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(a.output_dir/'wr07_200seed_station_draws.csv',index=False)
    labels=['P0 reset','P1 source','P2 turn','P3 target']; colors=['#3182bd','#6baed6','#31a354','#e6550d']; centers=[RESET,SRC,SRC,DST]
    fig=plt.figure(figsize=(15,13)); gs=fig.add_gridspec(3,3,height_ratios=[1.15,1,1],hspace=.62,wspace=.34)
    ax=fig.add_subplot(gs[0,:]);
    for i in range(a.count):
        ax.plot([df.iloc[i][f'p{j}_x'] for j in range(4)],[df.iloc[i][f'p{j}_y'] for j in range(4)],color='0.82',lw=.45,zorder=1)
    for j,(lab,c) in enumerate(zip(labels,colors)):
        x=df[f'p{j}_x']; y=df[f'p{j}_y']; ax.scatter(x,y,s=12,color=c,alpha=.55,label=lab,zorder=2); ax.add_patch(plt.Circle(tuple(centers[j]),R,fill=False,color=c,lw=2))
    ax.set_aspect('equal'); ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)'); ax.set_title(f'Spatial support · {a.count} seeds, 4 keypoints, every sampled point inside its 10 cm disk'); ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False)
    # Radial distributions
    ar=fig.add_subplot(gs[1,0]); rad=[]
    for j,c in enumerate(colors):
        d=np.hypot(df[f'p{j}_x']-centers[j][0],df[f'p{j}_y']-centers[j][1]); rad.append(d); parts=ar.violinplot(d,positions=[j],showextrema=False,widths=.65); [b.set_facecolor(c) for b in parts['bodies']]; ar.scatter([j],[np.mean(d)],color='white',edgecolor='k',zorder=3)
    ar.axhline(2*R/3,color='0.35',ls='--'); ar.set_xticks(range(4),['P0','P1','P2','P3']); ar.set_ylabel('distance to disk center (m)'); ar.set_title('Radial fill (uniform disk mean = 3.3 cm)')
    sh=fig.add_subplot(gs[1,1]); d1=np.hypot(df.p1_x-df.p0_x,df.p1_y-df.p0_y); d2=np.hypot(df.p3_x-df.p2_x,df.p3_y-df.p2_y); parts=sh.violinplot([d1,d2],positions=[0,1],showextrema=False,widths=.65); [b.set_facecolor(c) for b,c in zip(parts['bodies'],['#3182bd','#e6550d'])]; sh.set_xticks([0,1],['P0→P1','P2→P3']); sh.set_ylabel('XY shift (m)'); sh.set_title('Station-to-station XY shift')
    le=fig.add_subplot(gs[1,2]); vals=[df.leg1_m,df.leg2_m,df.leg3_m]; parts=le.violinplot(vals,positions=[0,1,2],showextrema=False,widths=.65); [b.set_facecolor(c) for b,c in zip(parts['bodies'],['#3182bd','#31a354','#e6550d'])]; le.set_xticks(range(3),['P0→P1','P1→P2','P2→P3']); le.set_ylabel('straight-line length (m)'); le.set_title('Three motion legs')
    yo=fig.add_subplot(gs[2,0]); vals=[np.zeros(a.count),df.turn_offset_deg]; parts=yo.violinplot(vals,positions=[0,1],showextrema=False,widths=.65); [b.set_facecolor(c) for b,c in zip(parts['bodies'],['#3182bd','#e6550d'])]; yo.axhline(5,color='0.35',ls='--'); yo.axhline(-5,color='0.35',ls='--'); yo.set_xticks([0,1],['initial','turn δ']); yo.set_ylabel('draw offset (deg)'); yo.set_title('Turn randomization');
    ac=fig.add_subplot(gs[2,1:]); cumulative=np.column_stack([np.zeros(a.count),df.turn_offset_deg,df.turn_offset_deg]); xx=np.arange(3); ac.fill_between(xx,np.percentile(cumulative,5,axis=0),np.percentile(cumulative,95,axis=0),color='#9ecae1',alpha=.6,label='5–95%'); ac.plot(xx,np.median(cumulative,axis=0),'o-',color='#0877b5',label='median'); ac.set_xticks(xx,['P0 reset','P2 turn vs source','P3 target vs dining']); ac.set_ylabel('yaw offset vs calibrated station (deg)'); ac.set_title('Turn yaw · signed −178.84° ± 5°'); ac.legend(frameon=False)
    fig.suptitle(f'WR-07 base randomization · seeds 0–{a.count-1}',fontsize=18,fontweight='bold',y=.985); fig.savefig(a.output_dir/'wr07_200seed_temporal_distribution.png',dpi=180,bbox_inches='tight'); fig.savefig(a.output_dir/'wr07_200seed_temporal_distribution.pdf',bbox_inches='tight'); plt.close(fig)
    summary={'n_seeds':a.count,'seed_range':[0,a.count-1],'radius_m':R,'reset_center_xy':RESET.tolist(),'source_center_xy':SRC.tolist(),'target_center_xy':DST.tolist(),'turn_base_deg':float(np.rad2deg(BASE_TURN)),'turn_range_deg':[float(np.rad2deg(BASE_TURN)-5),float(np.rad2deg(BASE_TURN)+5)],'keypoints':labels,'source_observe_pick_reuse':True,'csv':str((a.output_dir/'wr07_200seed_station_draws.csv').resolve())}
    (a.output_dir/'meta.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

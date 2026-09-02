"""Ordering ablation at n=100: co_p1 vs cte_p1 vs etc_p1 on the abl100 tile set.

Sources (all gemma-4-31b, temperature 1.0, replicate 1):
  cte : runs/cte_p1_full.jsonl for ALL 100 tiles (one consistent cte source,
        the same baselines the masking sweep used)
  co  : runs/co_p1.jsonl for the 18 tiles shared with the pilot 20,
        runs/co_p1_abl100.jsonl for the other 82
  etc : runs/etc_p1.jsonl / runs/etc_p1_abl100.jsonl likewise

Reports accuracy and SSA-call rate per condition, the paired cte->etc label
flips with an exact McNemar test (the p1 pilot found 6 HP->SSA vs 0 reverse),
and the cte<->co comparison (does producing an explanation change the label).

    python3 analyze_ordering100.py   ->  runs/ordering_abl100_analysis.json
"""
import json, math, os, sys
from collections import Counter
ROOT=os.path.dirname(os.path.abspath(__file__))
def load(p):
    out={}
    for l in open(os.path.join(ROOT,p)):
        if not l.strip(): continue
        r=json.loads(l)
        if r.get('replicate',1)==1 and (r.get('parsed') or {}).get('label') in ('HP','SSA'): out[r['image']]=r
    return out
def mcnemar(a,b):
    n=a+b
    if n==0: return 1.0
    return min(1.0, 2*sum(math.comb(n,i) for i in range(min(a,b)+1))/2**n)
def clopper(k,n,alpha=0.05):
    def bincdf(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    lo,hi=0.0,1.0
    if k>0:
        a,b=0.0,1.0
        for _ in range(200):
            m=(a+b)/2
            if 1-bincdf(k-1,n,m)>alpha/2: b=m
            else: a=m
        lo=a
    if k<n:
        a,b=0.0,1.0
        for _ in range(200):
            m=(a+b)/2
            if bincdf(k,n,m)<alpha/2: b=m
            else: a=m
        hi=a
    return round(lo,4),round(hi,4)

tiles=sorted(d+'.png' for d in os.listdir(os.path.join(ROOT,'masked','abl100_k3')) if d.startswith('MHIST_'))
cte_all=load('runs/cte_p1_full.jsonl')
co={**{k:v for k,v in load('runs/co_p1.jsonl').items()}, **load('runs/co_p1_abl100.jsonl')}
etc={**{k:v for k,v in load('runs/etc_p1.jsonl').items()}, **(load('runs/etc_p1_abl100.jsonl') if os.path.exists(os.path.join(ROOT,'runs/etc_p1_abl100.jsonl')) else {})}
conds={'co_p1':co,'cte_p1':cte_all,'etc_p1':etc}
rep={'n_tiles':len(tiles),'conditions':{},'pairs':{}}
for name,d in conds.items():
    have=[t for t in tiles if t in d]
    if not have: rep['conditions'][name]='MISSING'; continue
    k=sum(1 for t in have if d[t]['parsed']['label']==d[t]['label_true'])
    ssa=sum(1 for t in have if d[t]['parsed']['label']=='SSA')
    conf=[float(d[t]['parsed']['confidence'] or 0) for t in have]
    rep['conditions'][name]={'n':len(have),'accuracy':f'{k}/{len(have)}','acc_pct':round(100*k/len(have),1),'exact_95ci':clopper(k,len(have)),'ssa_calls':ssa,'ssa_rate':round(ssa/len(have),3),'mean_confidence':round(sum(conf)/len(conf),3)}
def pair(a,b,label):
    shared=[t for t in tiles if t in conds[a] and t in conds[b]]
    if not shared: rep['pairs'][label]='MISSING'; return
    A={t:conds[a][t]['parsed']['label'] for t in shared}; B={t:conds[b][t]['parsed']['label'] for t in shared}
    hp2ssa=[t for t in shared if A[t]=='HP' and B[t]=='SSA']; ssa2hp=[t for t in shared if A[t]=='SSA' and B[t]=='HP']
    rep['pairs'][label]={'n_shared':len(shared),f'{a}=HP->{b}=SSA':len(hp2ssa),f'{a}=SSA->{b}=HP':len(ssa2hp),'agreement':round(1-(len(hp2ssa)+len(ssa2hp))/len(shared),3),'mcnemar_p':round(mcnemar(len(hp2ssa),len(ssa2hp)),4),'flipped_tiles_HP_to_SSA':sorted(t[6:9] for t in hp2ssa),'flipped_tiles_SSA_to_HP':sorted(t[6:9] for t in ssa2hp)}
pair('cte_p1','etc_p1','cte_vs_etc (ordering effect)')
pair('co_p1','cte_p1','co_vs_cte (does explaining change the label)')
pair('co_p1','etc_p1','co_vs_etc')
out=os.path.join(ROOT,'runs','ordering_abl100_analysis.json')
json.dump(rep,open(out,'w'),indent=2); print(json.dumps(rep,indent=2)); print('->',out)

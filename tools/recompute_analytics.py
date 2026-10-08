#!/usr/bin/env python3
"""Recompute CAPA PPC and the eight analytics tabs from the Archives sheet.

Usage:
    python tools/recompute_analytics.py              # rewrites FBC_Data.xlsx in place
    python tools/recompute_analytics.py --out x.xlsx # write elsewhere

Run after tools/import_cup.py (and after the Handicaps and Cups columns for the new
cup exist). Uses every cup in Archives. Like import_cup.py it edits only the nine
analytics worksheet XML parts and copies everything else byte for byte; Excel
recalculates on open (fullCalcOnLoad is already set).

Methods (reverse-engineered from the FBC 12 snapshot in October 2026 and checked
against it; see the PR that added this file):
  Sandbagger  (Win% - 0.5) x mean Handicaps-tab index; min 10 matches (FTAS counted)
  Clutch      stake-weighted win% - flat win%; min 10 matches
  Chemistry   doubles tandem win% - mean of partners' career win%; min 4 together
  Pythagorean per-cup exp-2 expectation from official cup scores (OFFICIAL below;
              cups not listed use Archives team totals with the FTAS counted once);
              participation from the Cups sheet; min 4 cups
  Form Guide  last-3-cups PPC - career PPC; |delta| >= 0.5 is Hot/Cold; min 4 cups
  MVP         undefeated individual cups, min 3 matches
  Nemesis     head-to-head (singles + doubles opponents), min 3 meetings, players
              with 10+ matches; ranked by W-L, then W (or L), then meetings
  Streaks     non-FTAS matches in date order; ties break streaks; min 10 matches
  CAPA        (revised Oct 2026) stake-weighted ridge rating model over all head-to-head
              matches, no handicap term, lambda by leave-one-cup-out CV; reported as
              points in an FBC 13-format cup (3.25 + 4 x rating); min 24 matches
"""
import re, sys, zipfile, argparse
import numpy as np, pandas as pd
import openpyxl
from xml.sax.saxutils import escape
def load(path='/home/claude/fbc-data/FBC_Data.xlsx'):
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    rows=list(wb['Archives'].iter_rows(values_only=True))
    hdr=[h if h else f'c{i}' for i,h in enumerate(rows[0])]
    df=pd.DataFrame(rows[1:],columns=hdr)
    df=df[df['FBC'].notna()].copy()
    df['FBC']=df['FBC'].astype(int)
    df['row']=range(len(df))
    h=list(wb['Handicaps'].iter_rows(values_only=True))
    return df,h,wb
def players(r):
    return [p for p in (r['Player 1'],r['Player 2']) if isinstance(p,str) and p.strip()]
def long(df):
    out=[]
    for _,r in df.iterrows():
        for p in players(r):
            d=r.to_dict(); d['P']=p; out.append(d)
    return pd.DataFrame(out)


def hcp_table(h,N):
    hdr=h[2]; out={}
    for r in h[3:]:
        if not r[1] or r[1] in ('Average',) : continue
        vals=[]
        for j,c in enumerate(hdr):
            if isinstance(c,str) and c.startswith('FBC') and c!='FBC' and int(c[3:])<=N:
                v=r[j]
                if isinstance(v,(int,float)): vals.append(v)
        if vals: out[r[1]]=np.mean(vals)
    return out

def all_long(d):
    L=long(d); L['wv']=L['W']+0.5*L['T']; return L

def sandbagger(d,h,N):
    L=all_long(d); H=hcp_table(h,N)
    g=L.groupby('P').agg(n=('W','size'),wp=('wv','mean')).reset_index()
    g=g[(g.n>=10)&g.P.isin(H)]
    g['avg']=g.P.map(H); g['beat']=(g.wp-0.5)
    g['score']=(g.beat*g['avg']).round(2)
    g=g.sort_values(['score','P'],ascending=[False,True])
    return [(r.P,round(r.avg,1),int(r.n),round(r.wp,4),round(r.beat,4),r.score) for r in g.itertuples()]

def clutch(d):
    L=all_long(d); L['s']=L['Points at stake'].astype(float)
    L['sw']=L.s*L.wv
    g=L.groupby('P').agg(n=('W','size'),wp=('wv','mean'),sw=('sw','sum'),s=('s','sum')).reset_index()
    g=g[g.n>=10]; g['hl']=g.sw/g.s; g['d']=(g.hl-g.wp)
    g=g.sort_values('d',ascending=False)
    return [(r.P,int(r.n),round(r.wp,4),round(r.hl,4),round(r.d,4)) for r in g.itertuples()]

def chemistry(d):
    L=all_long(d); career=L.groupby('P').wv.mean()
    D=d[d['Singles/Doubles']=='Doubles'].copy()
    D['wv']=D['W']+0.5*D['T']
    D['tand']=D.apply(lambda r:' / '.join(sorted([r['Player 1'],r['Player 2']],key=str.lower)),axis=1)
    out=[]
    for t,g in D.groupby('tand'):
        if len(g)<4: continue
        a,b=t.split(' / '); e=(career[a]+career[b])/2; tw=g.wv.mean()
        out.append((t,len(g),round(tw,4),round(e,4),round(tw-e,4)))
    return sorted(out,key=lambda x:-x[4])

def cup_points(d):
    # team totals per cup, FTAS once
    res={}
    for f,g in d.groupby('FBC'):
        nf=g[g['Singles/Doubles']!='FTAS'].groupby('Team')['Points earned'].sum()
        ft=g[g['Singles/Doubles']=='FTAS'].groupby('Team')['Points earned'].max()
        tot=nf.add(ft,fill_value=0)
        res[f]=tot.to_dict()
    return res

def pythag(d):
    cp=cup_points(d); L=all_long(d)
    pt=L.groupby(['P','FBC']).Team.first().reset_index()
    out={}
    for r in pt.itertuples():
        t=cp[r.FBC]; pf=t.get(r.Team,0); pa=sum(v for k,v in t.items() if k!=r.Team)
        won=1 if pf>pa else 0; e=pf**2/(pf**2+pa**2)
        o=out.setdefault(r.P,[0,0,0.0]); o[0]+=1;o[1]+=won;o[2]+=e
    rows=[(p,c,w,round(e,2),round(w-e,2)) for p,(c,w,e) in out.items() if c>=4]
    return sorted(rows,key=lambda x:-x[4])

def form(d):
    L=all_long(d)
    pc=L.groupby(['P','FBC'])['Points earned'].sum().reset_index().sort_values('FBC')
    out=[]
    for p,g in pc.groupby('P'):
        if len(g)<4: continue
        car=g['Points earned'].mean(); l3=g['Points earned'].iloc[-3:].mean()
        dl=l3-car
        tr='▲  Hot' if dl>=0.5 else ('▼  Cold' if dl<=-0.5 else '—  Steady')
        out.append((p,len(g),round(car,2),round(l3,2),round(dl,2),tr))
    return sorted(out,key=lambda x:(-x[3],x[0]))

def mvp(d):
    L=all_long(d); out=[]
    for (p,f),g in L.groupby(['P','FBC']):
        if len(g)>=3 and g.L.sum()==0:
            out.append((p,f'FBC {f}',int(g.W.sum()),0,int(g['T'].sum()),float(g['Points earned'].sum())))
    out.sort(key=lambda x:(-x[5],-x[2],int(x[1][4:])))
    return [(i+1,)+x for i,x in enumerate(out)]

def h2h(d):
    rec={}
    H=d[d['Singles/Doubles']!='FTAS']
    for r in H.itertuples(index=False):
        r=r._asdict() if hasattr(r,'_asdict') else r
    for _,r in H.iterrows():
        me=[p for p in (r['Player 1'],r['Player 2']) if isinstance(p,str) and p]
        if r['Singles/Doubles']=='Singles': opp=[r['Singles Opponent']]
        else: opp=[o for o in (r['Opponent1'],r['Opponent2']) if isinstance(o,str) and o]
        for p in me:
            for o in opp:
                x=rec.setdefault((p,o),[0,0,0]); x[0]+=r.W; x[1]+=r.L; x[2]+=r['T']
    return rec

def nemesis(d,key='net'):
    rec=h2h(d); L=all_long(d); cnt=L.groupby('P').size()
    ps=sorted({p for p,_ in rec if cnt.get(p,0)>=10},key=str.lower); out=[]
    def sc(x):
        n=sum(x); return ((x[0]-x[1]), (x[0]+0.5*x[2])/n, n)
    for p in ps:
        c=[(o,x) for (q,o),x in rec.items() if q==p and sum(x)>=3]
        if not c: continue
        best=max(c,key=lambda z:(z[1][0]-z[1][1],z[1][0],sum(z[1]))); worst=max(c,key=lambda z:(z[1][1]-z[1][0],z[1][1],sum(z[1])))
        f=lambda x:f'{int(x[0])}–{int(x[1])}–{int(x[2])}'
        pa=(best[0],f(best[1])) if best[1][0]>best[1][1] else ('—',None)
        ne=(worst[0],f(worst[1])) if worst[1][1]>worst[1][0] else ('—',None)
        out.append((p,)+pa+ne)
    return out

def streaks(d,order='date'):
    H=d[d['Singles/Doubles']!='FTAS'].copy()
    H['mn']=pd.to_numeric(H['Match #'],errors='coerce')
    if order=='date': H=H.sort_values(['Date','mn','row'],kind='stable')
    else: H=H.sort_values(['FBC','row'],kind='stable')
    seq={}
    for _,r in H.iterrows():
        res='W' if r.W==1 else ('L' if r.L==1 else 'T')
        for p in (r['Player 1'],r['Player 2']):
            if isinstance(p,str) and p: seq.setdefault(p,[]).append(res)
    out=[]
    for p,s in seq.items():
        if len(s)<10: continue
        def lng(ch):
            b=c=0
            for x in s:
                c=c+1 if x==ch else 0; b=max(b,c)
            return b
        last=s[-1]; k=0
        for x in reversed(s):
            if x==last: k+=1
            else: break
        out.append((p,len(s),lng('W'),lng('L'),f'{last}{k}'))
    return sorted(out,key=lambda x:(-x[2],x[3],-x[1],x[0]))

# Official cup scores (winner, loser) per By-Laws Annex B; FBC 8 includes the later make-up singles (26-16.5); FBC 13 from Archives
OFFICIAL={1:(13,11),2:(18,14),3:(22,18),4:(22,18.5),5:(22.5,18),6:(22,18.5),7:(24.5,21),
          8:(26,16.5),9:(23,17.5),10:(23,22),11:(24,16.5),12:(28,22)}
def pythag2(wb,N,d=None):
    def score(f):
        if f in OFFICIAL: return OFFICIAL[f]
        t=cup_points(d[d.FBC==f])[f]; v=sorted(t.values(),reverse=True); return (v[0],v[1])
    cups=list(wb['Cups'].iter_rows(values_only=True)); hdr=cups[1]; out={}
    for r in cups[2:]:
        p=r[1]
        if not p or p=='Total': continue
        for j,c in enumerate(hdr):
            if isinstance(c,str) and c.startswith('FBC ') and int(c[4:])<=N and r[j] in (0,1):
                hi,lo=score(int(c[4:])); pf,pa=(hi,lo) if r[j]==1 else (lo,hi)
                o=out.setdefault(p,[0,0,0.0]); o[0]+=1;o[1]+=r[j];o[2]+=pf**2/(pf**2+pa**2)
    rows=[(p,c,w,round(e,2),round(w-e,2)) for p,(c,w,e) in out.items() if c>=4]
    return sorted(rows,key=lambda x:(-x[4],x[0]))

def hc_percup(h):
    hdr=h[2]; HC={}
    for r in h[3:]:
        if not r[1]: continue
        for j,c in enumerate(hdr):
            if isinstance(c,str) and c.startswith('FBC') and c[3:].isdigit() and isinstance(r[j],(int,float)): HC[(r[1],int(c[3:]))]=r[j]
    return HC
def skills(d,shrink=20,incl_ftas=True):
    L=all_long(d)
    if not incl_ftas: L=L[L['Singles/Doubles']!='FTAS']
    g=L.groupby('P').agg(n=('wv','size'),wp=('wv','mean'))
    return ((g.wp-0.5)*g.n/(g.n+shrink)).to_dict()

CAPA_LAMBDA=30      # ridge penalty; chosen by leave-one-cup-out cross-validation (Oct 2026)
CAPA_MIN_MATCHES=24 # matches incl. FTAS

def capa_matches(df,h):
    """One record per head-to-head match (FTAS excluded): sides A/B, A's share, stake."""
    X=df[df['Singles/Doubles']!='FTAS']; out=[]
    for mid,g in X.groupby('UniqueMatchID',sort=False):
        a,b=g.iloc[0],g.iloc[1]
        A=[p for p in (a['Player 1'],a['Player 2']) if isinstance(p,str) and p]
        B=[p for p in (b['Player 1'],b['Player 2']) if isinstance(p,str) and p]
        out.append(dict(FBC=int(a.FBC),A=A,B=B,stake=float(a['Points at stake']),share=a.W+0.5*a['T']))
    return pd.DataFrame(out)

def capa_fit(M,players,lam):
    """Stake-weighted ridge fit of  share_A - 0.5 = mean(r over A) - mean(r over B)."""
    ix={p:i for i,p in enumerate(players)}; X=np.zeros((len(M),len(players)))
    for k,r in enumerate(M.itertuples()):
        for p in r.A: X[k,ix[p]]+=1/len(r.A)
        for p in r.B: X[k,ix[p]]-=1/len(r.B)
    w=M.stake.values; y=M.share.values-0.5
    G=(X*w[:,None]).T@X+lam*np.eye(len(players))
    beta=np.linalg.solve(G,(X*w[:,None]).T@y)
    s2=(w*(y-X@beta)**2).sum()/w.sum()
    sd=np.sqrt(np.diag(s2*np.linalg.inv(G)))
    return {p:beta[i] for i,p in enumerate(players)},{p:sd[i] for i,p in enumerate(players)}

def capa_table(d,h,lam=CAPA_LAMBDA,min_matches=CAPA_MIN_MATCHES):
    """CAPA = 3.25 + 4 x rating: expected points in an FBC 13-format cup (four 1-pt doubles
    matches, one 2-pt singles, FTAS worth 0.25 on average) with an average partner against
    average opponents. Exact decomposition from the ridge normal equations:
      rating = (raw + partner + opponent) x V/(V+lam), V = sum of stake/c^2 (c = side size)."""
    M=capa_matches(d,h)
    players=sorted({p for x in M.A for p in x}|{p for x in M.B for p in x})
    r,sd=capa_fit(M,players,lam)
    L=all_long(d); nm=L.groupby('P').size(); cups=L.groupby('P').FBC.nunique()
    acc={}
    for x in M.itertuples():
        for side,opp,sh in ((x.A,x.B,x.share),(x.B,x.A,1-x.share)):
            c=len(side); v=x.stake/c**2
            for p in side:
                a=acc.setdefault(p,[0.0,0.0,0.0,0.0]); q=[t for t in side if t!=p]
                a[0]+=v; a[1]+=v*c*(sh-0.5); a[2]-=v*(r[q[0]] if q else 0.0); a[3]+=v*sum(r[o] for o in opp)
    out=[]
    for p,(V,u,P,O) in acc.items():
        if nm.get(p,0)<min_matches: continue
        u,P,O=u/V,P/V,O/V; adj=u+P+O; rr=adj*V/(V+lam)
        assert abs(rr-r[p])<1e-9
        out.append((p,int(cups[p]),int(nm[p]),round(3.25+4*u,2),round(4*P,2),round(4*O,2),round(4*(rr-adj),2),round(4*sd[p],2)))
    out.sort(key=lambda t:-(t[3]+t[4]+t[5]+t[6]))
    belts=dict(dragon=max(out,key=lambda t:t[5])[0],spoon=min(out,key=lambda t:t[4])[0],
               sisyphus=max(out,key=lambda t:t[4])[0],billy=min(out,key=lambda t:t[5])[0])
    return out,belts,len(M)

def all_tables(df,h,wb,N):
    d=df[df.FBC<=N]
    return {'Sandbagger Index':sandbagger(d,h,N),'Clutch Rating':clutch(d),'Chemistry Index':chemistry(d),
            'Pythagorean Luck':pythag2(wb,N,d),'Form Guide':form(d),'MVP Hall of Fame':mvp(d),
            'Nemesis & Patsy':nemesis(d),'Streak Tracker':streaks(d,'date'),'CAPA PPC':capa_table(d,h)}

def main():
    import os
    ap=argparse.ArgumentParser(); ap.add_argument('--out')
    a=ap.parse_args()
    here=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    global SRC
    SRC=os.path.join(here,'FBC_Data.xlsx'); OUT=a.out or SRC
    run(SRC,OUT)

def run(SRC,OUT):

    df,h,wb=load(SRC)
    N=int(df.FBC.max()); P=N-1
    T=all_tables(df,h,wb,N)
    z=zipfile.ZipFile(SRC); parts={n:z.read(n) for n in z.namelist()}; infos=z.infolist()
    wbx=parts['xl/workbook.xml'].decode(); rels=parts['xl/_rels/workbook.xml.rels'].decode()
    ss=re.findall(r'<si>(.*?)</si>',parts['xl/sharedStrings.xml'].decode(),re.S)
    def sstext(i): return ''.join(re.findall(r'<t[^>]*>([^<]*)</t>',ss[i]))
    def path(name):
        n=escape(name)
        rid=re.search(r'<sheet name="%s" sheetId="\d+"(?: state="\w+")? r:id="(rId\d+)"'%re.escape(n),wbx).group(1)
        t=(re.search(r'Id="%s"[^>]*Target="([^"]+)"'%rid,rels) or re.search(r'Target="([^"]+)"[^>]*Id="%s"'%rid,rels)).group(1)
        return 'xl/'+t.lstrip('/').replace('xl/','')
    COLS='ABCDEFGHIJ'
    def cell(ref,v,s):
        if v is None: return f'<c r="{ref}" s="{s}"/>'
        if isinstance(v,str): return f'<c r="{ref}" s="{s}" t="inlineStr"><is><t xml:space="preserve">{escape(v)}</t></is></c>'
        v=float(v); v=int(v) if v.is_integer() else v
        return f'<c r="{ref}" s="{s}"><v>{v}</v></c>'
    def rowcells(x): return dict(re.findall(r'<c r="([A-Z]+)\d+" s="(\d+)"',x))
    def set_text(s,ref,text):
        col,rn=re.match(r'([A-Z]+)(\d+)',ref).groups()
        m=re.search(r'<c r="%s"( s="\d+")?[^>]*?(?:/>|>.*?</c>)'%ref,s,re.S)
        st=m.group(1) or ''
        return s.replace(m.group(0),f'<c r="{ref}"{st} t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>',1)
    def cell_text(s,ref):
        m=re.search(r'<c r="%s"[^>]*?t="s"[^>]*><v>(\d+)</v>'%ref,s); return sstext(int(m.group(1))) if m else None

    def rewrite_table(name,start,first_col,data,special=None):
        p=path(name); s=parts[p].decode()
        allrows=[(int(m.group(2)),m.group(1)) for m in re.finditer(r'(<row r="(\d+)"[^>]*>.*?</row>|<row r="(\d+)"[^>]*/>)',s,re.S) if m.group(2)]
        old=[(n,x) for n,x in allrows if n>=start]
        tA,tB=old[0][1],old[1][1]; head=re.match(r'<row r="\d+"([^>]*)>',tA).group(1)
        sA,sB=rowcells(tA),rowcells(tB)
        # trend style map (Form Guide)
        tmap={}
        if special=='form':
            for i,(n,x) in enumerate(old):
                v=re.search(r'<c r="G%d" s="(\d+)" t="s"><v>(\d+)</v>'%n,x)
                if v: tmap.setdefault((sstext(int(v.group(2))),i%2),v.group(1))
                v=re.search(r'<c r="G%d" s="(\d+)" t="inlineStr"><is><t[^>]*>([^<]*)</t>'%n,x)
                if v: tmap.setdefault((v.group(2),i%2),v.group(1))
        new=[]
        for i,vals in enumerate(data):
            rn=start+i; sty=sA if i%2==0 else sB; cells=[]
            for j,v in enumerate(vals):
                col=COLS[COLS.index(first_col)+j]; st=sty[col]
                if special=='form' and col=='G': st=tmap.get((v,i%2),st)
                cells.append(cell(f'{col}{rn}',v,st))
            new.append(f'<row r="{rn}"{head}>'+''.join(cells)+'</row>')
        blk=''.join(x for _,x in old)
        assert blk in s
        s=s.replace(blk,''.join(new))
        last=start+len(data)-1
        s=re.sub(r'<dimension ref="([A-Z]+\d+):([A-Z]+)\d+"/>',lambda m:f'<dimension ref="{m.group(1)}:{m.group(2)}{last}"/>',s)
        parts[p]=s.encode(); return p

    def fmt(v):
        import numpy as np
        return float(v) if isinstance(v,(np.floating,)) else v
    def clean(rows): return [tuple(fmt(v) for v in r) for r in rows]

    for name in ['Sandbagger Index','Clutch Rating','Chemistry Index','Pythagorean Luck','Streak Tracker','Nemesis & Patsy']:
        rewrite_table(name,5,'B',clean(T[name]))
    rewrite_table('Form Guide',5,'B',clean(T['Form Guide']),special='form')
    rewrite_table('MVP Hall of Fame',5,'B',clean(T['MVP Hall of Fame']))
    p=path('Form Guide'); s=parts[p].decode()
    t=cell_text(s,'B2') or str(wb['Form Guide']['B2'].value); s=set_text(s,'B2',re.sub(r'into FBC \d+',f'into FBC {N+1}',t)); parts[p]=s.encode()

    # CAPA: rebuild rows 3-7 (method notes), 9 (header), the table, and the belts block
    capa,belts,nmatch=T['CAPA PPC']
    p=path('CAPA PPC'); s=parts[p].decode()
    rowsxml=re.findall(r'(<row r="(\d+)"[^>]*?(?:/>|>.*?</row>))',s,re.S)
    R={int(n):x for x,n in rowsxml}
    def rowtext(x): return ''.join(re.findall(r'<t[^>]*>([^<]*)</t>',x))+''.join(sstext(int(v)) for v in re.findall(r't="s"><v>(\d+)</v>',x))
    bh=[n for n,x in R.items() if 'Belts (through' in rowtext(x)][0]
    belt_rows=[R[bh+k] for k in range(1,5)]
    hdr=R[9]; dat=R[10]; dstyle=rowcells(dat); rattr=re.match(r'<row r="\d+"([^>]*)>',dat).group(1)
    notes={3:('What it measures',f'Expected points in a standard FBC 13-format cup (four 1-pt doubles matches, one 2-pt singles match, FTAS) with an average partner against average opponents. Through FBC{N}.'),
           4:('Model',f'Joint rating fit to all {nmatch} head-to-head matches (FTAS excluded): side A share − 0.5 = mean(A ratings) − mean(B ratings), weighted by points at stake, ridge-regularized (λ={CAPA_LAMBDA}, chosen by leave-one-cup-out cross-validation). CAPA = 3.25 + 4 × rating.'),
           5:('Columns','Raw = own results only, in standard-cup points. Partner / Opponent Adj = removes the help or hurt from partners’ and opponents’ ratings. Shrinkage = pull toward average for players with fewer matches. CAPA = Raw + Partner + Opponent + Shrinkage.'),
           6:('Handicaps','No separate handicap term. Strokes already level matches: adding an opponent-minus-own index term made out-of-sample predictions worse, so it was dropped (Oct 2026 revision).'),
           7:('Precision',f'The signal is weak: the model predicts match results about 2% better than a coin flip (the previous win%-based CAPA was worse than a coin flip at predicting the next cup). ± = one standard deviation; gaps smaller than that are noise. Min {CAPA_MIN_MATCHES} matches.')}
    for rn,(lab,txt_) in notes.items():
        s=set_text(s,f'B{rn}',lab); s=set_text(s,f'C{rn}',txt_)
        ht=15*max(2,-(-len(txt_)//70))   # merged C:J is ~75 characters wide
        s=re.sub(r'(<row r="%d"[^>]*?) ht="[\d.]+"'%rn,lambda m:f'{m.group(1)} ht="{ht}"',s,count=1)
    R={int(n):x for x,n in re.findall(r'(<row r="(\d+)"[^>]*?(?:/>|>.*?</row>))',s,re.S)}
    # Top-align the note labels (B3:B7) next to their wrapped text: reuse or add one cell style.
    sty=parts['xl/styles.xml'].decode(); cx=re.search(r'<cellXfs count="(\d+)">(.*?)</cellXfs>',sty,re.S)
    xfs=re.findall(r'<xf [^>]*?(?:/>|>.*?</xf>)',cx.group(2),re.S)
    lab_s=re.search(r'<c r="B3" s="(\d+)"',s).group(1)
    want=re.sub(r'/>$','',xfs[int(lab_s)]).replace('<alignment vertical="top"/></xf>','')
    want=want.rstrip('>') if want.endswith('>') and not want.endswith('/>') else want
    top=f'{want} applyAlignment="1"><alignment vertical="top"/></xf>' if 'vertical="top"' not in xfs[int(lab_s)] else xfs[int(lab_s)]
    if top in xfs: top_i=xfs.index(top)
    else:
        top_i=len(xfs)
        sty=sty.replace(cx.group(0),f'<cellXfs count="{top_i+1}">'+cx.group(2)+top+'</cellXfs>')
        parts['xl/styles.xml']=sty.encode()
    for rn in range(3,8): s=re.sub(r'<c r="B%d" s="\d+"'%rn,f'<c r="B{rn}" s="{top_i}"',s)
    s=s.replace('<col min="3" max="3" width="6" customWidth="1"/>','<col min="3" max="3" width="11" customWidth="1"/>')
    heads=['Player','Cups','Matches','Raw PPC','Partner Adj','Opponent Adj','Shrinkage','CAPA PPC','± (1 SD)']
    newhdr=re.sub(r'<c r="([A-J])9"( s="\d+")[^>]*?(?:/>|>.*?</c>)',lambda m:f'<c r="{m.group(1)}9"{m.group(2)} t="inlineStr"><is><t xml:space="preserve">{escape(heads["BCDEFGHIJ".index(m.group(1))])}</t></is></c>',hdr)
    st={c:dstyle[c] for c in 'BCDEFGHIJ'}; numst=dstyle['E']
    body=[]
    for i,t in enumerate(clean(capa)):
        rn=10+i; c=[cell(f'B{rn}',t[0],st['B']),cell(f'C{rn}',t[1],st['C']),cell(f'D{rn}',t[2],st['C'])]
        for col,v in zip('EFGH',t[3:7]): c.append(cell(f'{col}{rn}',v,numst))
        c.append(f'<c r="I{rn}" s="{st["I"]}"><f>E{rn}+F{rn}+G{rn}+H{rn}</f></c>')
        c.append(cell(f'J{rn}',t[7],numst))
        body.append(f'<row r="{rn}"{rattr}>'+''.join(c)+'</row>')
    last=9+len(capa); b0=last+2
    def renum(x,old,new): return re.sub(r'(r="[A-J]?)%d"'%old,lambda m:f'{m.group(1)}{new}"',x)
    bx=[renum(R[bh],bh,b0)]
    for k,(x,key) in enumerate(zip(belt_rows,('dragon','spoon','sisyphus','billy'))):
        y=renum(x,bh+1+k,b0+1+k); y=set_text(y,f'C{b0+1+k}',belts[key]); bx.append(y)
    bx[0]=set_text(bx[0],f'B{b0}',f'Belts (through FBC{N})')
    keep=''.join(R[n] for n in sorted(R) if n<9)
    a0=s.index('<sheetData>')+len('<sheetData>'); a1=s.index('</sheetData>')
    s=s[:a0]+keep+newhdr+''.join(body)+f'<row r="{last+1}"{rattr}/>'+''.join(bx)+s[a1:]
    for rn in range(3,8): s=re.sub(r'<c r="B%d" s="\d+"'%rn,f'<c r="B{rn}" s="{top_i}"',s)
    s=re.sub(r'<mergeCell ref="D\d+:J\d+"/>','',s)
    mc=''.join(f'<mergeCell ref="D{b0+k}:J{b0+k}"/>' for k in range(1,5))
    s=re.sub(r'<mergeCells count="\d+">',lambda m:'<mergeCells count="9">'+mc,s)
    s=re.sub(r'<conditionalFormatting sqref="[HJ]\d+:[HJ]\d+">.*?</conditionalFormatting>','',s,flags=re.S)
    s=re.sub(r'<conditionalFormatting sqref="([FGI])10:[FGI]\d+">',lambda m:f'<conditionalFormatting sqref="{m.group(1)}10:{m.group(1)}{last}">',s)
    s=re.sub(r'<dimension ref="[^"]+"/>',f'<dimension ref="B1:J{b0+4}"/>',s)
    parts[p]=s.encode()
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as w:
        for i in infos: w.writestr(i,parts[i.filename])
    print('wrote',OUT)

if __name__=='__main__':
    main()

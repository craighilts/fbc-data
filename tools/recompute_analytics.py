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
  CAPA        fixed coefficients (1.757 skill, 0.0078 handicap) from the FBC12 fit;
              skill = (career win% - 0.5) x n/(n+20); min 25 matches; FBC 8 credit
              = stake/6.5 for players who missed the make-up singles
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
OFFICIAL={1:(9.5,8.5),2:(18,14),3:(22,18),4:(22,18.5),5:(22.5,18.5),6:(22,18.5),7:(24.5,21),
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

def capa(d,h,bs=1.757,bh=0.0078,shrink=20,incl_ftas=True,pmode='half',omode='mean',hmode='mean',eff8=4.5/6.5,partial8=None):
    sk=skills(d,shrink,incl_ftas); HC=hc_percup(h)
    acc={}
    L=all_long(d)
    for _,r in d.iterrows():
        me=[p for p in (r['Player 1'],r['Player 2']) if isinstance(p,str) and p]
        sd=r['Singles/Doubles']; st=float(r['Points at stake'])
        if sd=='FTAS':
            for p in me:
                a=acc.setdefault(p,dict(pts=0,pa=0,oa=0,ha=0,cups=set())); a['pts']+=r['Points earned']; a['cups'].add(r.FBC)
            continue
        op=[r['Singles Opponent']] if sd=='Singles' else [o for o in (r['Opponent1'],r['Opponent2']) if isinstance(o,str) and o]
        hk=all((x,r.FBC) in HC for x in me+op)
        for p in me:
            a=acc.setdefault(p,dict(pts=0,pa=0,oa=0,ha=0,cups=set())); a['pts']+=r['Points earned']; a['cups'].add(r.FBC)
            q=[x for x in me if x!=p]
            pc=(sk.get(q[0],0)/2 if pmode=='half' else sk.get(q[0],0)) if q else 0
            oc=np.mean([sk.get(o,0) for o in op]) if omode=='mean' else np.sum([sk.get(o,0) for o in op])
            a['pa']+= -st*bs*pc
            a['oa']+= st*bs*oc
            if hk:
                if hmode=='mean': hv=np.mean([HC[(o,r.FBC)] for o in op])-np.mean([HC[(x,r.FBC)] for x in me])
                elif hmode=='self': hv=np.mean([HC[(o,r.FBC)] for o in op])-HC[(p,r.FBC)]
                else: hv=np.sum([HC[(o,r.FBC)] for o in op])-np.sum([HC[(x,r.FBC)] for x in me])
                a['ha']+= -st*bh*hv
    return acc,sk

def capa_table(d,h,min_matches=25):
    acc,sk=capa(d,h)
    L=all_long(d); n=L.groupby('P').size()
    st8=L[L.FBC==8].groupby('P')['Points at stake'].sum()
    out=[]
    for p,a in acc.items():
        if n.get(p,0)<min_matches: continue
        cups=len(a['cups']); eff=cups-(1-min(1,st8[p]/6.5) if p in st8 else 0)
        r=(p,cups,round(eff,2),round(a['pts']/eff,4),round(a['pa']/eff,4),round(a['oa']/eff,4),round(a['ha']/eff,4))
        out.append(r)
    out.sort(key=lambda r:-(r[3]+r[4]+r[5]+r[6]))
    belts=dict(dragon=max(out,key=lambda r:r[5])[0],spoon=min(out,key=lambda r:r[4])[0],
               sisyphus=max(out,key=lambda r:r[4])[0],billy=min(out,key=lambda r:r[5])[0])
    return out,belts
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

    # CAPA: fixed 23 rows (10-32), keep formula cells, drop cached values
    capa,belts=T['CAPA PPC']; assert len(capa)==23
    p=path('CAPA PPC'); s=parts[p].decode()
    for i,r in enumerate(clean(capa)):
        rn=10+i; x=re.search(r'<row r="%d"[^>]*>.*?</row>'%rn,s,re.S).group(0); y=x
        for j,col in enumerate('BCDEFGH'):
            m=re.search(r'<c r="%s%d" s="(\d+)"[^>]*?(?:/>|>.*?</c>)'%(col,rn),y,re.S)
            y=y.replace(m.group(0),cell(f'{col}{rn}',r[j],m.group(1)),1)
        y=re.sub(r'(<c r="[IJ]%d"[^>]*>(?:<f[^>]*>[^<]*</f>|<f[^>]*/>))<v>[^<]*</v>'%rn,r'\1',y)
        s=s.replace(x,y,1)
    for ref,key in (('C35','dragon'),('C36','spoon'),('C37','sisyphus'),('C38','billy')): s=set_text(s,ref,belts[key])
    def txt(ref): return cell_text(s,ref) or str(wb['CAPA PPC'][ref].value)
    for ref,pat,rep in (('C3',r'Through FBC\d+\.',f'Through FBC{N}.'),('C6',r'FBC1–\d+',f'FBC1–{N}'),('B34',r'through FBC\d+',f'through FBC{N}')):
        s=set_text(s,ref,re.sub(pat,rep,txt(ref)))
    t=re.sub(r' Coefficients carried forward.*$','',txt('C4'))
    s=set_text(s,'C4',t+' Coefficients carried forward from the FBC12 fit; not re-estimated since.')
    parts[p]=s.encode()
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as w:
        for i in infos: w.writestr(i,parts[i.filename])
    print('wrote',OUT)

if __name__=='__main__':
    main()

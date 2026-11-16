import pandas as pd, numpy as np, networkx as nx, itertools
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
P=pd.read_parquet('data/processed/passes.parquet'); A=pd.read_parquet('data/processed/appearances.parquet'); M=pd.read_parquet('data/processed/matches.parquet')
P=P[(P.team_id==217)&P.completed].merge(M[['match_id','season','era']]); A=A[A.team_id==217].merge(M[['match_id','season','era']])
# units
mins=A.groupby(['season','player_id']).minutes_played.sum()
pg=A.groupby(['season','player_id','position_group']).minutes_played.sum().reset_index().sort_values('minutes_played').groupby(['season','player_id']).tail(1).set_index(['season','player_id']).position_group
U=pd.DataFrame({'mins':mins,'grp':pg}); U=U[(U.mins>=900)&(U.grp!='GK')]
# centrality per match
cent=[]
for mid,a in A.groupby('match_id'):
    p=P[P.match_id==mid]; p=p[p.player_id.isin(a.player_id)&p.recipient_id.isin(a.player_id)&(p.player_id!=p.recipient_id)]
    G=nx.DiGraph(); G.add_nodes_from(a.player_id)
    for (u,v),n in p.groupby(['player_id','recipient_id']).size().items(): G.add_edge(u,v,weight=n,dist=1/n)
    pr=nx.pagerank(G,weight='weight'); b=nx.betweenness_centrality(G,weight='dist',normalized=True); n=G.number_of_nodes()
    for r in a.itertuples(): cent.append((mid,r.season,r.player_id,r.minutes_played,pr[r.player_id]*n,b[r.player_id]))
C=pd.DataFrame(cent,columns=['match_id','season','player_id','m','pr','bt'])
C['wpr']=C.pr*C.m; C['wbt']=C.bt*C.m; c=C.groupby(['season','player_id'])[['wpr','wbt','m']].sum()
F=pd.DataFrame(index=U.index); F['pagerank_n']=c.wpr/c.m; F['betweenness']=c.wbt/c.m
made=P.groupby(['season','player_id']); recv=P.groupby(['season','recipient_id'])
F['made_p90']=made.size().reindex(F.index)/U.mins*90; F['recv_p90']=recv.size().rename_axis(['season','player_id']).reindex(F.index)/U.mins*90
def cell(x,y): return (np.digitize(x,[40,80])).astype(str)+(np.digitize(y,[80/3,160/3])).astype(str)
P['mc']=cell(P.x,P.y); P['rc']=cell(P.end_x,P.end_y)
for kind,col,key in [('made','mc','player_id'),('recv','rc','recipient_id')]:
    t=P.groupby(['season',key,col]).size().unstack(fill_value=0); t=t.div(t.sum(1),axis=0); t.index.names=['season','player_id']
    for cc in t.columns: F[f'{kind}_{cc}']=t[cc].reindex(F.index).fillna(0)
P['prog']=(P.x>=48)&(np.hypot(120-P.end_x,40-P.end_y)<=0.75*np.hypot(120-P.x,40-P.y)); P['elig']=P.x>=48
g=P.groupby(['season','player_id'])
F['prog_share']=(g.prog.sum()/g.elig.sum()).reindex(F.index).fillna(0)
F['len']=g.apply(lambda d: np.hypot(d.end_x-d.x,d.end_y-d.y).mean()).reindex(F.index)
F['fwd']=g.apply(lambda d:(d.end_x>d.x).mean()).reindex(F.index)
print('rows',len(F), 'features', F.shape[1], 'NaN', F.isna().sum().sum())
X=StandardScaler().fit_transform(F.values); pca=PCA().fit(X); cum=np.cumsum(pca.explained_variance_ratio_); nc=int(np.argmax(cum>=0.8)+1)
Z=PCA(nc).fit_transform(X); print('evr',pca.explained_variance_ratio_[:6].round(3),'kept',nc)
labs={}
for k in range(3,7):
    km=KMeans(k,n_init=50,random_state=671).fit(Z); labs[k]=km.labels_; print(k, round(silhouette_score(Z,km.labels_),4))
L=labs[6]
R=pd.read_parquet('data/processed/o2_roles.parquet')
key=R.set_index(['season','player_id'])
print('ARI mine vs pipeline', round(adjusted_rand_score(L, key.reindex(F.index)['cluster']),4), 'ARI vs positions', round(adjusted_rand_score(L,U.grp),4))
# era role mix exact test
F['cl']=L; F['mins']=U.mins
seasons=sorted(F.index.get_level_values(0).unique())
def tv(pep):
    s=F.reset_index(); s['g']=s.season.isin(pep)
    sh=s.groupby(['g','cl']).mins.sum().unstack(fill_value=0); sh=sh.div(sh.sum(1),axis=0); return 0.5*np.abs(sh.loc[True]-sh.loc[False]).sum()
obs=tv(['2008/2009','2009/2010','2010/2011','2011/2012']); allv=[tv(list(c)) for c in itertools.combinations(seasons,4)]
print('TV obs',round(obs,4),'p',sum(v>=obs-1e-12 for v in allv)/35)

# --- compare features to pipeline, then test row-order sensitivity
PF=pd.read_parquet('data/processed/player_season_features.parquet').set_index(['season','player_id'])
feat=[c for c in PF.columns if PF[c].dtype!=object and c not in ('minutes','era')][-25:]
print('pipeline feature cols', len(feat))
Fm=F.drop(columns=['cl','mins']); Pm=PF.loc[Fm.index, feat]
print('max abs diff (sorted columns, by position):', np.abs(np.sort(Fm.values,0)-np.sort(Pm.values,0)).max().round(6))
Xp=StandardScaler().fit_transform(Pm.values); Zp=PCA(4).fit_transform(Xp)
for order in ['pipeline order','my order','reversed','shuffled']:
    idx=np.arange(len(Zp))
    if order=='reversed': idx=idx[::-1]
    if order=='shuffled': idx=np.random.default_rng(0).permutation(len(Zp))
    km=KMeans(6,n_init=50,random_state=671).fit(Zp[idx]); lab=np.empty(len(Zp),int); lab[idx]=km.labels_
    print(order, 'sil', round(silhouette_score(Zp,lab),4), 'ARI vs pipeline', round(adjusted_rand_score(lab, key.loc[Pm.index,'cluster']),3))
for seed in range(5):
    km=KMeans(6,n_init=50,random_state=seed).fit(Zp); print('seed',seed, round(silhouette_score(Zp,km.labels_),4), round(km.inertia_,1), 'ARI', round(adjusted_rand_score(km.labels_, key.loc[Pm.index,'cluster']),3))

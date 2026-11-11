"""Partition/sequence/lexical diagnostics using stored parser outputs and review histories."""
from cpu_research_common import *
import re,csv
from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jensenshannon

def tokens(s):return re.findall('[a-z]+',(s or '').replace('<*>',' ').lower())
def partition(gt,pred):
 c=Counter(zip(gt,pred));a=Counter(gt);b=Counter(pred);pair=lambda n:n*(n-1)//2
 tp=sum(pair(n) for n in c.values());pp=sum(pair(n) for n in b.values());gp=sum(pair(n) for n in a.values())
 byp=defaultdict(Counter);byg=defaultdict(Counter)
 for (g,p),n in c.items():byp[p][g]=n;byg[g][p]=n
 return {'ARI':adjusted_rand_score(gt,pred),'NMI':normalized_mutual_info_score(gt,pred),'pair_precision':tp/pp if pp else 1.,'pair_recall':tp/gp if gp else 1.,'predicted_groups':len(b),'GT_groups':len(a),'merged_groups':sum(len(v)>1 for v in byp.values()),'split_GT_events':sum(len(v)>1 for v in byg.values()),'merge_affected_lines':sum(sum(v.values()) for v in byp.values() if len(v)>1),'split_affected_lines':sum(sum(v.values()) for v in byg.values() if len(v)>1),'event_macro_split_fraction':np.mean([len(v)>1 for v in byg.values()]),'pred_event_macro_purity':np.mean([max(v.values())/sum(v.values()) for v in byp.values()]),'GT_event_macro_completeness':np.mean([max(v.values())/sum(v.values()) for v in byg.values()])}
def mapping(gt,pred):
 c=defaultdict(Counter)
 for g,p in zip(gt,pred):c[p][g]+=1
 m={p:sorted(v,key=lambda g:(-v[g],g))[0] for p,v in c.items()}
 return m,{'ambiguous_pred_events':sum(len(v)>1 for v in c.values()),'tied_pred_events':sum(list(v.values()).count(max(v.values()))>1 for v in c.values()),'unmatched_events':0}
def js(a,b):
 keys=sorted(set(a)|set(b),key=str)
 if not keys:return None
 x=np.array([a[k] for k in keys],float);y=np.array([b[k] for k in keys],float)
 if not x.sum() or not y.sum():return None
 return float(jensenshannon(x/x.sum(),y/y.sum(),base=2)**2)
def seqstats(gt,pred,units):
 m,info=mapping(gt,pred);mp=[m[p] for p in pred];idx=[i for u in units for i in u['lines']]
 out={**info,'mapped_changed_positions':float(np.mean([gt[i]!=mp[i] for i in idx])),'positions':len(idx),'vocabulary_ratio':len(set(pred))/len(set(gt))}
 for n in (1,2,3):
  def counts(v):
   z=Counter()
   for u in units:
    s=[v[i] for i in u['lines']];z.update(tuple(s[j:j+n]) for j in range(len(s)-n+1))
   return z
  out[f'ngram{n}_JS_bits']=js(counts(gt),counts(mp))
 # ngram2 is the transition joint distribution; no cross-unit edges.
 return out

def lexstats(gt,pred,indices,vectorizers):
 pairs=Counter((gt[i],pred[i]) for i in indices);keys=list(pairs);w=np.array(list(pairs.values()),float)
 out={};sim0=None
 for name,v in vectorizers.items():
  x=v.transform([g for g,p in keys]);y=v.transform([p or '' for g,p in keys]);sim=np.asarray(x.multiply(y).sum(1)).ravel();both=np.asarray(x.getnnz(1)==0)&np.asarray(y.getnnz(1)==0);sim[both]=1
  out[name+'_cosine']=float(np.average(sim,weights=w));out[name+'_zero_GT_pairs']=int(sum(x.getnnz(1)==0));
  if name=='word':sim0=sim
 lost=[];added=[];overlap=[]
 for g,p in keys:
  a=Counter(tokens(g));b=Counter(tokens(p));lost.append(sum((a-b).values()));added.append(sum((b-a).values()));overlap.append(sum((a&b).values())/max(1,sum(a.values())))
 out.update(lost_constants_per_line=float(np.average(lost,weights=w)),unsupported_constants_per_line=float(np.average(added,weights=w)),constant_recall=float(np.average(overlap,weights=w)))
 pergt=defaultdict(list)
 for k,(g,p) in enumerate(keys):pergt[g].append(k)
 out['word_cosine_GT_event_macro']=float(np.mean([np.average(sim0[ii],weights=w[ii]) for ii in pergt.values()]))
 freq=Counter(gt);rare=[i for i,(g,p) in enumerate(keys) if freq[g]<=20];frequent=[i for i,(g,p) in enumerate(keys) if freq[g]>20]
 for name,ii in [('rare_le20',rare),('frequent_gt20',frequent)]:out[name+'_word_cosine']=float(np.average(sim0[ii],weights=w[ii])) if ii else None
 return out,[{'GT':g,'prediction':p,'lines':pairs[g,p],'word_cosine':float(sim0[i]),'lost_constants':lost[i],'unsupported_constants':added[i]} for i,(g,p) in enumerate(keys)]
def main():
 allrows=[];pairrows=[];validation={}
 for system in ('HDFS','BGL'):
  d=load(EXP/'data/stream'/f'{system}.json');r=load(EXP/'data/strong_downstream_preflight'/f'{system}_representations.json.gz');units=load(EXP/'data/strong_downstream_preflight'/f'{system}_units.json.gz');gt=r['conditions']['GT'];conditions=dict(r['conditions']);groups={k:d['group'] for k in conditions if k!='GT'};histories={}
  for model,suf in [('llama',''),('gemma','_gemma')]:
   strong=load(EXP/'data/stream_gen_strong'/model/'C2'/f'{system}.json');print(system,model,'strong keys',list(strong),flush=True)
   cache=strong['final'];conditions[model+'/strong']=[cache[g] for g in d['group']];groups[model+'/strong']=d['group']
   for parser,name in [('base',model+'/cache'),('strong',model+'/strong')]:
    hp=EXP/f'results_phase3{suf}/robustness_tail'/parser/f'{system}.json'
    fallback=EXP/f'results_phase3{suf}'/('triage/e1/base' if parser=='base' else 'strong')/f'{system}.json'
    h=load(hp if hp.exists() else fallback);histories[name]=dict(h['full']['runs'] if 'full' in h else h['runs'])
    zp=EXP/f'results_phase3{suf}/zero_target/streams'/parser/f'{system}.json'
    if zp.exists():histories[name].update(load(zp)['review'])
   h=load(EXP/f'results_phase3{suf}/adaparser_e2e'/f'{system}.json');histories[model+'/AdaParser']=h['runs']
   # Real AdaParser groups are assigned in first-occurrence order, as its simulator.
   ids={};groups[model+'/AdaParser']=[ids.setdefault(t,len(ids)) for t in conditions[model+'/AdaParser']]
  train=[i for u in units if u['split']=='train' for i in u['lines']];test=[i for u in units if u['split']=='test' for i in u['lines']]
  assert set(train).isdisjoint(test)
  # Shared vocabulary across parser conditions, training rows only. GT inclusion is diagnostic.
  docs=sorted({p[i] or '' for p in conditions.values() for i in train})
  vec={'word':TfidfVectorizer(tokenizer=tokens,token_pattern=None,lowercase=False).fit(docs),'char':TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1).fit(docs)}
  save(DATA/f'{system}_lexical_fit.json',{'training_line_ids':[r['line_ids'][i] for i in train],'vocabularies':{k:v.vocabulary_ for k,v in vec.items()},'idf':{k:v.idf_.tolist() for k,v in vec.items()},'diagnostic_GT_vocabulary':True})
  validation[system]={'lines':len(gt),'units':len(units),'shared_training_documents':len(docs),'source_sha256':sha(EXP/'data/strong_downstream_preflight'/f'{system}_representations.json.gz')}
  for name,pred in conditions.items():
   assert len(pred)==len(gt)==50000
   operational=groups.get(name,gt)
   baseline=partition(gt,operational)
   for method,seed,budget,rv in [('none',0,0,[])]+[(m,s,b,list(histories.get(name,{}).get(m,{}).get(str(s),{}).get('seq',[]))[:b]) for m in ('frequent','dsize','usize','U-zero x size','U+V5-zero x size','oracle') if m in histories.get(name,{}) for s in range(5) for b in (5,10,20,50)]:
    if method!='none' and not rv:continue
    assert len(rv)==len(set(rv));c=defaultdict(Counter)
    for i,g in enumerate(operational):c[g][gt[i]]+=1
    corrections={g:sorted(v,key=lambda x:(-v[x],x))[0] for g,v in c.items()}
    rvset=set(rv)
    pp=[corrections[g] if g in rvset else pred[i] for i,g in enumerate(operational)]
    lex,pairs=lexstats(gt,pp,test,vec)
    z={'system':system,'parser':name,'method':method,'seed':seed,'budget':budget,'actual_reviews':len(rv),'corrected_lines':sum(pp[i]==gt[i] and pred[i]!=gt[i] for i in range(len(gt))),'changed_template_groups':sum(any(pred[i]!=pp[i] for i,g0 in enumerate(operational) if g0==g) for g in rv),'partition_scope':'operational assignments fixed under template-only review','operational':baseline,'string_partition':partition(gt,pp),'sequence':seqstats(gt,pp,units),'test_lexical':lex}
    allrows.append(z)
    if budget==0:
     pairrows.extend({'system':system,'parser':name,**x} for x in pairs)
   save(OUT/f'representation_{system}.json',allrows_for_system(allrows,system))
 save(OUT/'representation_validation.json',validation)
 save(OUT/'lexical_pairs.json',pairrows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,8))
 base=[z for z in allrows if z['budget']==0 and z['parser']!='GT'];names=[z['system']+'/'+z['parser'] for z in base];x=np.arange(len(base))
 axs[0,0].plot(x,[z['string_partition']['ARI'] for z in base],'o',label='ARI');axs[0,0].plot(x,[z['string_partition']['NMI'] for z in base],'x',label='NMI');axs[0,0].legend()
 axs[0,1].bar(x,[z['sequence']['mapped_changed_positions'] for z in base]);axs[0,1].set_title('Majority-GT mapped position change (oracle diagnostic)')
 for ax in axs[0]:ax.set_xticks(x,names,rotation=90,fontsize=6)
 for name in sorted({z['parser'] for z in base}):
  zz=[z for z in pairrows if z['system']=='BGL' and z['parser']==name];axs[1,0].hist([z['word_cosine'] for z in zz],weights=[z['lines'] for z in zz],bins=20,histtype='step',label=name)
 axs[1,0].set_title('BGL test line-weighted lexical cosine');axs[1,0].legend(fontsize=6)
 for name in ['llama/cache','llama/AdaParser','gemma/AdaParser']:
  for method in ['frequent','dsize','usize']:
   zz=[z for z in allrows if z['system']=='BGL' and z['parser']==name and z['method']==method]
   if zz:
    bs=sorted({z['budget'] for z in zz});axs[1,1].plot(bs,[np.mean([z['test_lexical']['word_cosine'] for z in zz if z['budget']==b]) for b in bs],label=name+'/'+method)
 axs[1,1].set_title('BGL test lexical recovery; invalid anomaly split');axs[1,1].legend(fontsize=6)
 fig.tight_layout();fig.savefig(OUT/'representation.pdf');plt.close(fig)
 print('rows',len(allrows),flush=True)
def allrows_for_system(rows,s):return [z for z in rows if z['system']==s]
if __name__=='__main__':main()

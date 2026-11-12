"""Exact unlimited-review representation ceilings from existing group memberships."""
from cpu_research_common import *
from collections import Counter,defaultdict
import numpy as np,csv
from sklearn.metrics import adjusted_rand_score
ROOT=EXP/'results_phase3/cpu_remaining';ROOT.mkdir(parents=True,exist_ok=True)

def analyze(gt,groups,text):
 assert len(gt)==len(groups)==len(text);n=len(gt);ct=defaultdict(Counter);grouptext={}
 for g,t,p in zip(groups,gt,text):
  ct[g][t]+=1
  if g in grouptext:assert grouptext[g]==p,'one text per operational group required'
  grouptext[g]=p
 mixed={g for g,c in ct.items() if len(c)>1};mixedtypes={t for g in mixed for t in ct[g]};gttypes=set(gt);bygt=defaultdict(set)
 for g,c in ct.items():
  for t in c:bygt[t].add(g)
 major={g:sorted(c,key=lambda t:(-c[t],t))[0] for g,c in ct.items()};corrected=[major[g] for g in groups];right=sum(p==t for p,t in zip(text,gt));maxright=sum(max(c.values()) for c in ct.values());gross=sum(p!=t and q==t for p,q,t in zip(text,corrected,gt));broken=sum(p==t and q!=t for p,q,t in zip(text,corrected,gt));residual=n-maxright
 assert maxright-right==gross-broken and sum(q!=t for q,t in zip(corrected,gt))==residual
 pairs=lambda k:k*(k-1)//2
 intrinsic=sum(pairs(sum(c.values()))-sum(pairs(x) for x in c.values()) for c in ct.values());before_bad_pairs=sum(pairs(v) for v in Counter(groups).values())-sum(pairs(v) for c in ct.values() for v in c.values())
 assert intrinsic==before_bad_pairs
 # This intervention optimizes exact line accuracy, not necessarily ARI.
 cid={};gid={};corrids=[cid.setdefault(v,len(cid)) for v in corrected];truthids=[gid.setdefault(v,len(gid)) for v in gt]
 post_ARI=adjusted_rand_score(truthids,corrids)
 gtcounts=Counter(gt);goodtypes=gttypes-mixedtypes
 result={'lines':n,'GT_types':len(gttypes),'operational_groups':len(ct),'currently_wrong_text_lines':n-right,'current_line_accuracy':right/n,'pure_groups':len(ct)-len(mixed),'mixed_groups':len(mixed),'mixed_group_lines':sum(sum(ct[g].values()) for g in mixed),'pure_wrong_text_lines':sum(p!=t and g not in mixed for g,t,p in zip(groups,gt,text)),'mixed_wrong_text_lines':sum(p!=t and g in mixed for g,t,p in zip(groups,gt,text)),
 'mixed_GT_types':len(mixedtypes),'split_GT_types':sum(len(v)>1 for v in bygt.values()),'pure_split_GT_types':sum(len(v)>1 and t not in mixedtypes for t,v in bygt.items()),'merge_touched_line_share':sum(gtcounts[t] for t in mixedtypes)/n,
 'exact_max_correct_lines_one_string_per_group':maxright,'exact_line_accuracy_ceiling':maxright/n,'gross_newly_correct_lines':gross,'newly_wrong_lines':broken,'net_correct_line_gain':maxright-right,'exact_unavoidable_wrong_lines':residual,
 'exact_min_member_moves_after_free_relabel_and_coalescing':residual,'necessary_group_fragment_increase':sum(len(c)-1 for c in ct.values()),'exact_simultaneously_restorable_GT_types':len(goodtypes),'exact_restorable_GT_event_lines':sum(gtcounts[t] for t in goodtypes),'irreducible_cross_GT_pairs_within_original_groups':intrinsic,
 'line_optimal_string_coalescing_ARI':float(post_ARI),'operational_partition_change_under_text_only':False,'oracle_is_not_deployable':True,'no_anomaly_detection_ceiling':True,'observed_parse_failure_lines':sum(t is None for t in text)}
 result['ko_calculation']=f"고정 그룹마다 가장 많은 정답 템플릿의 빈도를 합산하면 {maxright}/{n}줄이 맞으며, 최소 {residual}줄은 한 그룹당 한 문자열만으로 복구할 수 없다."
 result['ko_meaning']=f"혼합 그룹 {len(mixed)}개가 정답 이벤트 {len(mixedtypes)}종을 건드린다. 문자열 교정 후 동일 문자열을 합칠 때 완전히 복원 가능한 정답 이벤트는 {len(goodtypes)}종이다."
 result['ko_limitation']='정답을 이용한 무제한 교정의 표현 수준 상한이다. 실제 검토 정책의 성능이나 이상 탐지 F1 상한이 아니며, 원래 운영 그룹 ID는 텍스트 교정만으로 바뀌지 않는다.'
 return result,ct

def main():
 rows=[];groups=[];sources={}
 for p in sorted((EXP/'data/stream').glob('*.json')):
  d=load(p)
  if not isinstance(d,dict) or 'gt' not in d:continue
  system=p.stem;gt=[d['templates'][t] for t in d['gt']];op=d['group'];n=len(gt);sources[str(p.relative_to(EXP))]=sha(p)
  dp=EXP/'data/stream_drain_templates'/p.name;dr=load(dp);assert dr['group']==op;conditions=[('shared','Drain',op,[dr['templates'][g] for g in op])]
  for model in ('llama','gemma'):
   for parser in ('base','strong'):
    sp=EXP/'data/severity_audit'/model/f'stream_{parser}_{system}.json.gz'
    if not sp.exists():continue
    grows=load(sp);cache={r['group']:r['predicted'] for r in grows};assert len(cache)==len(set(op));pred=[cache[g] for g in op];conditions.append((model,parser,op,pred));sources[str(sp.relative_to(EXP))]=sha(sp)
   ap=EXP/'data/adaparser_e2e'/model/p.name
   if ap.exists():
    a=load(ap);pred=a['templates'];assert len(pred)==n and a['provenance']['version']==1;ids={};gr=[ids.setdefault(t,len(ids)) for t in pred];conditions.append((model,'real_AdaParser',gr,pred));sources[str(ap.relative_to(EXP))]=sha(ap)
  for model,parser,gr,text in conditions:
   result,ct=analyze(gt,gr,text);result={'system':system,'model':model,'parser':parser,'covered_LineId_start':d['line_start'],'covered_LineId_end':d['line_start']+n-1,'missing_full_horizon_coverage':'Not counted as observed error; only saved prefix analyzed',**result};rows.append(result)
   if model!='shared':
    audit=load(EXP/f'results_phase3{"" if model=="llama" else "_gemma"}/severity_audit'/p.name)['streams'][('real_adaparser' if parser=='real_AdaParser' else parser)]
    assert audit['operational_partition']['mixed_pred_groups']==result['mixed_groups'];assert audit['operational_partition']['gt_groups']==result['GT_types']
    grows=load(EXP/'data/severity_audit'/model/f'stream_{"real_adaparser" if parser=="real_AdaParser" else parser}_{system}.json.gz')
    assert sum(r['net_gain'] for r in grows)==result['net_correct_line_gain']
   for g,c in ct.items():
    if len(c)>1:groups.append({'system':system,'model':model,'parser':parser,'group':int(g),'members_by_GT':dict(c),'unavoidable_wrong_lines':sum(c.values())-max(c.values()),'ko':'이 그룹에는 여러 정답 이벤트가 섞여 있어 단일 템플릿 교정으로 모든 줄을 동시에 맞출 수 없다.'})
  save(ROOT/'correction_opportunity.json',rows)
 save(ROOT/'correction_mixed_groups.json.gz',groups);save(ROOT/'correction_sources.json',sources)
 with (ROOT/'correction_opportunity.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(12,5));systems=sorted({r['system'] for r in rows})
 for model in ('llama','gemma'):
  z=[r for r in rows if r['model']==model and r['parser']=='real_AdaParser'];ss=[r['system'] for r in z];xx=[systems.index(s) for s in ss];axs[0].plot(xx,[100*r['exact_unavoidable_wrong_lines']/r['lines'] for r in z],'o-',label=model);axs[1].plot(xx,[r['mixed_GT_types']/r['GT_types'] for r in z],'o-',label=model)
  axs[0].set_xticks(range(len(systems)),systems,rotation=90,fontsize=7);axs[1].set_xticks(range(len(systems)),systems,rotation=90,fontsize=7)
 axs[0].set_ylabel('Unavoidable wrong lines (%)');axs[1].set_ylabel('Fraction GT types requiring a group split')
 for ax in axs:ax.legend();ax.set_title('Actual AdaParser; saved coverage only')
 fig.tight_layout();fig.savefig(ROOT/'correction_opportunity.pdf');plt.close(fig)
 print('exact correction ceilings',len(rows),'conditions; mixed groups',len(groups))
if __name__=='__main__':main()

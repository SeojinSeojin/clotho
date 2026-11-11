"""Score-blind parsing-error proxies; no semantic/harm ground truth is inferred."""
import re
from collections import Counter
from functools import lru_cache
import regex


def words(t):return Counter(re.findall('[a-z]+',(t or '').lower().replace('<*>',' ')))


@lru_cache(20000)
def pattern(t,greedy):
    p=t.split('<*>');return regex.compile(''.join(regex.escape(x)+('(.*)' if greedy else '(.*?)') for x in p[:-1])+regex.escape(p[-1]))


def align(t,line):
    out=[]
    for greedy in (False,True):
        try:m=pattern(t,greedy).fullmatch(line,timeout=.003)
        except TimeoutError:return None,'timeout'
        if not m:return None,'nonmatch'
        spans=[m.span(i) for i in range(1,len(m.groups())+1)]
        out.append(spans)
    if out[0]!=out[1]:return None,'ambiguous'
    mask=bytearray(len(line))
    for a,b in out[0]:mask[a:b]=b'\1'*(b-a)
    return (mask,out[0]),None


@lru_cache(200000)
def classify(content,pred,gt):
    r={'primary':'uncertain','strict':pred!=gt,'structural':None,'retained_variable':False,'masked_constant':False,
       'slot_change':False,'word_changed':words(pred)!=words(gt),'unsupported_words_proxy':False,'semantic_verified':None,'mask_signature':None}
    if pred==gt:
        r.update(primary='exact',structural=False,mask_signature=['masked']*gt.count('<*>'));return r
    if pred is not None and ' '.join(pred.split())==' '.join(gt.split()):
        r.update(primary='surface_whitespace_proxy',structural=False,mask_signature=['masked']*gt.count('<*>'));return r
    if not pred:r.update(primary='nonmatching_representation_proxy',structural=True,reason='parse_failure');return r
    line=' '.join(content.split());g=' '.join(gt.split());p=' '.join(pred.split())
    ga,ge=align(g,line)
    if ge:r['reason']='GT_'+ge;return r
    pa,pe=align(p,line)
    if pe:
        if pe=='nonmatch':
            r.update(primary='nonmatching_representation_proxy',structural=True,reason='prediction_nonmatch')
            r['unsupported_words_proxy']=bool(set(words(p))-set(re.findall('[a-z]+',line.lower())))
        else:r['reason']='prediction_'+pe
        return r
    gm,gs=ga;pm,ps=pa
    r['retained_variable']=any(a and not b and not c.isspace() for a,b,c in zip(gm,pm,line))
    r['masked_constant']=any(not a and b and not c.isspace() for a,b,c in zip(gm,pm,line))
    r['slot_change']=len(gs)!=len(ps)
    r['mask_signature']=['masked' if all(pm[a:b]) else 'retained' if not any(pm[a:b]) else 'partial' for a,b in gs]
    r['structural']=bool(r['retained_variable'] or r['masked_constant'] or r['slot_change'])
    r['primary']='variable_extraction_proxy' if r['structural'] else 'uncertain'
    if not r['structural']:r['reason']='same_roles_other_template_difference'
    return r


def majority(v):
    return True if sum(x is True for x in v)>=3 else False if sum(x is False for x in v)>=3 else None

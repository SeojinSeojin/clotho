"""G5 V2: strict stored-token alignment, with explicit unknown/missing values."""
import re
import numpy as np
from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template, _TEMPLATE_LINE


def spans(raw):
    t,err=parse_result_messages_template(raw or '')
    if err:return None
    start=None;pos=0
    for line in raw.split('\n'):
        if _TEMPLATE_LINE.match(line):start=pos
        pos+=len(line)+1
    i=raw.find(t,start)
    if i<0:return None
    wild=[(i+m.start(),i+m.end()) for m in re.finditer(r'<\*>',t)]
    digit=[(i+m.start(),i+m.end()) for m in re.finditer(r'\S+',t) if '<*>' not in m.group() and any(c.isdigit() for c in m.group())]
    return wild,digit


def indices(offsets,ss,shift=0,token_start=0):
    out=[]
    for group in ss:
        ids=[]
        for a,b in group:
            found=[j-token_start for j,(x,y) in enumerate(offsets) if y>x and y>a+shift and x<b+shift and j>=token_start]
            if not found:return None
            ids.append(found[0])
        out.append(tuple(ids))
    return tuple(out)


def summarize(lp,idx):
    if idx is None:return [0.,0.,1.,0.,0.,1.,1.]
    out=[]
    for ids in idx:
        x=np.asarray(lp,dtype=float)[list(ids)]
        if not len(x):out.extend([0.,0.,1.])
        else:out.extend([float(x.mean()),float(x.min()),0.])
    return out+[0.]


def suite(raw,lp,tok,chat):
    ss=spans(raw)
    if ss is None:return summarize(lp,None)
    prefix=tok.encode(chat,add_special_tokens=False)
    enc=tok(chat+raw,add_special_tokens=False,return_offsets_mapping=True)
    if enc.input_ids[:len(prefix)]!=prefix or len(lp)!=len(enc.input_ids)-len(prefix)+1:return summarize(lp,None)
    idx=indices(enc.offset_mapping,ss,len(chat),len(prefix))
    if idx is not None and any(i<0 or i>=len(lp)-1 for g in idx for i in g):idx=None
    return summarize(lp,idx)


def stream(raw,lp,tok):
    if not raw or lp is None:return summarize([],None)
    ss=spans(raw)
    if ss is None:return summarize(lp,None)
    candidates=set()
    for pre in ('',' ','\n','\n\n'):
        for post in ('',' ','\n','\n\n'):
            text=pre+raw+post
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            if len(lp)-len(enc.input_ids) not in (0,1):continue
            if tok.decode(enc.input_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text:continue
            idx=indices(enc.offset_mapping,ss,len(pre))
            if idx is not None:candidates.add(idx)
    return summarize(lp,next(iter(candidates)) if len(candidates)==1 else None)

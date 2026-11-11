import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from token_decision_features import spans,indices,summarize


def test_final_answer_decisions_exclude_reasoning_and_variable_digits():
    raw='Template: ignored 111\nReasoning: 999\nTemplate: x<*> a12 <*>:34 z'
    s=spans(raw)
    assert [raw[a:b] for a,b in s[0]]==['<*>','<*>']
    assert [raw[a:b] for a,b in s[1]]==['a12']
    idx=indices([(i,i+1) for i in range(len(raw))],s)
    lp=list(range(len(raw)))
    v=summarize(lp,idx)
    assert v[0]==sum(a for a,b in s[0])/2
    assert v[1]==min(a for a,b in s[0]) and v[-1]==0


def test_missing_alignment_and_no_decisions_are_distinct():
    assert summarize([],None)==[0.,0.,1.,0.,0.,1.,1.]
    assert summarize([], ((),()))==[0.,0.,1.,0.,0.,1.,0.]

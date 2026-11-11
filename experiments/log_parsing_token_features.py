"""G5 CPU V2 feature extraction from saved sample0/greedy outputs only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments';sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP/'sharded'))
import model_cfg as MC
import token_decision_features as V
OUT=EXP/'data/token_followups'/MC.MODEL


def features(push):
    import pandas as pd
    from transformers import AutoTokenizer
    import log_parsing_unseen as U
    import log_parsing_phase3_lib as L
    import log_parsing_triage as T
    import log_parsing_strong_sim as SS
    import prompt_render as PR
    from clotho.exp_config import task2template
    tok=AutoTokenizer.from_pretrained(MC.CFG['hf_id'],local_files_only=True,use_fast=True)
    assert tok.is_fast
    OUT.mkdir(parents=True,exist_ok=True)
    for ds in sorted(U.capped()):
        system=U.SYSNAME(ds);p=OUT/f'{system}_v2.json'
        if p.exists():continue
        t0=time.time();src=[json.loads(x)['content'] for x in (L.SUITES/f'{ds}.jsonl').read_text().splitlines()]
        gen=json.loads((EXP/f'data/results_{MC.MODEL}/log_parsing/inference_results/rules_template/{ds}_R10_T0.8.json').read_text())
        scores=pd.read_pickle(EXP/f'data/results_{MC.MODEL}/log_parsing/precalculated_metrics/rules_template/{ds}_postgen_samples.pkl')
        scores=scores[scores.inference_index==0].sort_values('input_index')
        assert scores.input_index.tolist()==list(range(len(src)))
        values=[]
        for text,g,lp in zip(src,gen,scores.token_logprobs):
            msgs=task2template['log_parsing']['rules_template'](text)
            if MC.MODEL=='gemma':msgs=PR.inline_system_as_user(msgs)
            chat=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            values.append(V.suite(g['inferences'][0],lp,tok,chat))
        r={'model':MC.MODEL,'system':system,'suite':values}
        for parser,g in [('base',T.gen(system)),('strong',SS.strong_gen(system))]:
            r[parser]=[V.stream(raw,lp,tok) for raw,lp in zip(g['greedy_raw'],g['token_logprobs'])]
        r['coverage']={k:float(np.mean(np.asarray(r[k])[:,-1]==0)) for k in ('suite','base','strong')}
        p.write_text(json.dumps(r));print(system,r['coverage'],f'{time.time()-t0:.1f}s',flush=True)
        if push:
            from sharded import hub_sync as hs
            hs.commit_files(hs.get_results_repo(),{str(p.relative_to(EXP)):p},f'G5 V2 {MC.MODEL} {system}')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--push',action='store_true');features(a.parse_args().push)

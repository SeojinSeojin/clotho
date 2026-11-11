"""G5 fixed token pooling; exact HF prompts, no generation, per-system persistence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments';sys.path.insert(0,str(EXP))
import model_cfg as MC
OUT=EXP/'data/token_followups'/MC.MODEL
LAYERS=[24] if MC.MODEL=='llama' else [21,28]


def extract(push):
    import torch
    import log_parsing_unseen as U
    import log_parsing_phase3_lib as L
    import log_parsing_stream as ST
    import log_parsing_triage as T
    from clotho.exp_config import task2template
    np.random.seed(0);torch.manual_seed(0)
    w=MC.wrapper();w.model.eval();tok=w.tokenizer;fn=task2template['log_parsing']['rules_template']
    OUT.mkdir(parents=True,exist_ok=True)
    for ds in sorted(U.capped()):
        system=U.SYSNAME(ds);p=OUT/f'{system}.npz';meta=p.with_suffix('.json')
        if p.exists() and meta.exists():continue
        src=L.SUITES/f'{ds}.jsonl';rows=[json.loads(x)['content'] for x in src.read_text().splitlines()]
        stream=ST.load(system)['first_content']
        arrays={};parity={};t0=time.time()
        for kind,contents in [('suite',rows),('stream',stream)]:
            values={f'{kind}_{pool}_{layer}':[] for layer in LAYERS for pool in ('mean','max')}
            lengths=[]
            for i,text in enumerate(contents):
                msgs=MC.chat_messages(fn(text));prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                inputs=tok(prompt,return_tensors='pt',add_special_tokens=MC.MODEL=='llama').to(w.device)
                with torch.inference_mode():
                    hs=w.model.model(**inputs,output_hidden_states=True,use_cache=False).hidden_states
                    if i==0:
                        full=w.get_hidden_states(text,fn,produce_inference_results=False)['hidden_states']
                        for layer in LAYERS:
                            assert torch.equal(hs[layer],full[layer]),(system,kind,layer,'base/wrapper parity')
                            stored=U.hidden(ds,layer)[0] if kind=='suite' else np.asarray(T.states(system,layer))[0]
                            actual=hs[layer][0,-1].float().cpu().numpy()
                            assert np.array_equal(actual,stored),(system,kind,layer,'stored last token parity',float(np.max(abs(actual-stored))))
                            parity[f'{kind}_{layer}']=True
                        del full
                    for layer in LAYERS:
                        H=hs[layer][0][inputs.attention_mask[0].bool()].float()
                        values[f'{kind}_mean_{layer}'].append(H.mean(0).cpu().numpy())
                        values[f'{kind}_max_{layer}'].append(H.max(0).values.cpu().numpy())
                    lengths.append(int(inputs.attention_mask.sum()))
                    del hs,H,inputs
                if (i+1)%250==0:print(system,kind,i+1,len(contents),f'{time.time()-t0:.0f}s',flush=True)
            arrays.update({k:np.asarray(v,dtype=np.float32) for k,v in values.items()});arrays[kind+'_lengths']=np.asarray(lengths)
        np.savez_compressed(p,**arrays)
        meta.write_text(json.dumps({'model':MC.MODEL,'system':system,'ds':ds,'layers':LAYERS,'pool_dtype':'float32',
            'suite_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'parity':parity,
            'suite_n':len(rows),'stream_n':len(stream),'wall_s':time.time()-t0,'seed':0},indent=1))
        print(system,'DONE',len(rows),len(stream),f'{time.time()-t0:.0f}s',flush=True)
        if push:
            from sharded import hub_sync as hs
            hs.commit_files(hs.get_results_repo(),{str(q.relative_to(EXP)):q for q in (p,meta)},f'G5 pooled states {MC.MODEL} {system}')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--push',action='store_true');extract(a.parse_args().push)

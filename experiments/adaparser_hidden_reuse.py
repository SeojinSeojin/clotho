"""Provisional CPU replay using exact stored zero-shot first-line prompt matches.

Every vector is subsequently re-extracted by the prescribed GPU hidden step.
"""
import json
from pathlib import Path
import numpy as np


def reuse(E,system,push=False):
    from log_parsing_adaparser_review import source_hash
    source=E.DATA/f'{system}.json'
    if not source.exists():return False
    meta=json.loads(source.read_text())
    if meta.get('provenance',{}).get('version')!=1:return False
    layer=24 if E.MODEL=='llama' else 35
    dest=E.DATA/f'{system}_layer{layer}.npy';side=dest.with_suffix('.meta.json')
    digest=source_hash(source)
    if dest.exists() and side.exists() and json.loads(side.read_text()).get('source_sha256')==digest:return False
    cached=E.EXP/f'data/stream_hidden/{E.MODEL}/{system}_layer{layer}.npy'
    if not cached.exists():return False
    stream=E.ST.load(system);rows=E.ST.read_stretch(system);_,first,_=E.groups(system)
    lookup={}
    for i,text in enumerate(stream['first_content']):lookup.setdefault(text,i)
    texts=[rows[i][1] for i in first]
    if not all(t in lookup for t in texts):return False
    H=np.load(cached,mmap_mode='r');assert len(H)==len(stream['first_content'])
    indices=[lookup[t] for t in texts]
    np.save(dest,np.asarray(H[indices],np.float32))
    side.write_text(json.dumps({'source_sha256':digest,'model':E.MODEL,'layer':layer,'n':len(indices),
        'source_method':'exact_stream_reuse','gpu_verified':False,
        'stream_file':str(cached.relative_to(E.EXP)),'stream_sha256':source_hash(cached),'stream_indices':indices}))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(p.relative_to(E.EXP)):p for p in (dest,side)},f'Provisional exact hidden reuse {E.MODEL} {system}')
    print('Provisional exact hidden reuse',system,len(indices),flush=True)
    return True

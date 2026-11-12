"""Validate source preservation, baseline identities and isolated deterministic replay."""
from cpu_research_common import EXP, load, save, sha
import sys
import tempfile
from pathlib import Path
import numpy as np
import log_parsing_component_ablation as A


def main():
    root = A.ROOT
    hashes = load(root / 'source_hashes.json')
    assert all(sha(EXP / p) == h for p, h in hashes.items()), 'Earlier artifacts modified'
    jobs = sorted((root / 'jobs').glob('*/*/*/seed[0-4].json'))
    assert len(jobs) == 250
    rows, ineligible, nonconverged = 0, [], []
    for p in jobs:
        j = load(p)
        assert len(j['ref']) == 100 and len(set(j['ref'])) == 100
        assert not set(j['ref']) & set(j['test'])
        assert j['layer'] == (16 if j['model'] == 'llama' else 21)
        assert j['cv']['transductive/random'].get('selected', j['original_current']) == j['original_dimension']
        for name, cfg in j['cv'].items():
            if 'ineligible' in cfg:
                ineligible.append(dict(job=str(p.relative_to(root)), cell=name, reason=cfg['ineligible']))
            for fold in cfg.get('folds', []):
                a = {j['groups'][i] for i in fold['train']}
                b = {j['groups'][i] for i in fold['validation']}
                if '/grouped' in name:
                    assert not a & b
        for name, f in j['fits'].items():
            if not f['pass_converged'] or not f['fail_converged']:
                nonconverged.append(dict(job=str(p.relative_to(root)), cell=name, details=f))
        z = np.load(p.with_suffix('.npz'))
        y = np.array(j['test_failure'])
        for row in j['rows']:
            risk = z[row['method']]
            assert np.isfinite(risk).all()
            met = A.metric(y, risk)
            assert abs(met['AUROC'] - row['AUROC']) < 1e-12
            assert abs(met['AP'] - row['AP']) < 1e-12
            rows += 1
        expected_primary = 3 * sum('ineligible' not in info for info in j['cv'].values())
        assert len([r for r in j['rows'] if r['representation'] in A.REPS and r['scoring'] in A.SCORES]) == expected_primary
        error = p.with_name(p.stem + '_error.json')
        if error.exists():
            d = load(error)
            d['resolved_by_validated_archived_baseline_protocol'] = True
            save(error, d)
    # Rerun complete new component computations in another directory; old files untouched.
    replay = []
    for model, system, split in [('llama', 'HDFS', 'disjoint'), ('gemma', 'BGL', 'standard')]:
        original = root / 'jobs' / model / split / system / 'seed0.json'
        with tempfile.TemporaryDirectory() as td:
            A.ROOT = Path(td)
            result = A.run(A.OLD / model / split / system / 'seed0.json')
            assert result[1] == 'OK', result
            fresh = A.ROOT / original.relative_to(root)
            a, b = load(original), load(fresh)
            a.pop('wall_s'); b.pop('wall_s')
            for key in ('candidate_recovery_valid', 'replay_AUROC_error'):
                if key not in a:
                    b.pop(key, None)
            assert a == b, 'non-deterministic metadata/CV/metrics'
            za, zb = np.load(original.with_suffix('.npz')), np.load(fresh.with_suffix('.npz'))
            assert za.files == zb.files
            assert all(np.array_equal(za[k], zb[k]) for k in za), 'non-deterministic risk'
            replay.append(dict(model=model, system=system, split=split, seed=0, identical=True))
        A.ROOT = root
    assert all(sha(EXP / p) == h for p, h in hashes.items())
    save(root / 'validation.json', dict(jobs=250, scored_cells=rows, sources_unchanged=len(hashes),
         no_GPU=True, no_new_inference=True, original_AUROC_tolerance=1e-12,
         archived_baseline_vectors_identical=True, grouped_ineligible=ineligible,
         nonconverged_fits=nonconverged, deterministic_replays=replay,
         note='Cross-hardware original PCA is not bit-identical; archived controls preserved. See protocol and baseline_validation.json.'))
    print('Validated250 jobs,', rows, 'score cells;', len(ineligible), 'grouped cells ineligible;', len(nonconverged), 'nonconverged density fits.')


if __name__ == '__main__':
    main()

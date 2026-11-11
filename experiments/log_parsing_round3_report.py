"""Live R1-R3 report, reading only completed/resumable CPU artifacts."""
import json
import math
import statistics
from pathlib import Path
import numpy as np
from log_parsing_stats import sk_esd, a12
E = Path(__file__).resolve().parent
REP = E / 'reports'
MODELS = {'llama': '', 'gemma': '_gemma'}
SC = ('strict', 'L1', 'L2')
SEEDS = range(5)
TUNE = {'hdfs', 'spark', 'linux'}


def held(s):
    return s.lower().replace('loghub_', '').replace('_nat500', '').replace('_500', '') not in TUNE


def fmt(v):
    return f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}' if len(v) == 5 else 'n/a'


def stats(values, reference):
    """All methods on common systems/seeds; no different-system paired comparisons."""
    methods = [m for m, ss in values.items() if ss]
    if not methods or reference not in methods:
        return {}
    systems = sorted(set.intersection(*(set(values[m]) for m in methods)))
    systems = [s for s in systems if all(len(values[m][s]) == 5 and
               all(v is not None and math.isfinite(v) for v in values[m][s]) for m in methods)]
    if not systems:
        return {}
    obs = {m: [v for s in systems for v in values[m][s]] for m in methods}
    per = {m: np.mean([values[m][s] for s in systems], axis=0) for m in methods}
    ranks = sk_esd(obs)
    out = {}
    for m in methods:
        dd = per[m]-per[reference]
        out[m] = {'per_seed': per[m].tolist(), 'text': fmt(per[m]), 'delta': fmt(dd),
                  'delta_mean': float(dd.mean()), 'positive_seeds': int(sum(dd > 0)),
                  'sign_seeds': int(sum(x != 0 and (x > 0) == (dd.mean() > 0) for x in dd)),
                  'rank': ranks[m], 'a12': a12(obs[m], obs[reference]), 'systems': systems}
    return out


def table(detail, title, result, ref):
    detail += ['', '### '+title, '', '| method | mean ± std | minus '+ref+' | positive seeds | SK-ESD | A12 | systems |',
               '|---|---|---|---:|---:|---:|---:|']
    for m, r in result.items():
        detail.append(f"| {m} | {r['text']} | {r['delta']} | {r['positive_seeds']}/5 | {r['rank']} | {r['a12']:.3f} | {len(r['systems'])} |")


def read(path):
    return json.loads(path.read_text())


def completed_cells(root):
    out = {}
    for parser in ('base', 'strong'):
        out[parser] = {p.stem: read(p) for p in (root / parser).glob('*.json') if read(p).get('complete')}
    return out


def suite_stats(root, model, detail, methods, target='clotho', strict_u=False):
    all_results = {}
    for kind in ('disjoint', 'standard', 'natural'):
        paths = sorted((root / kind).glob('*/seed*.json'))
        jobs = {}
        for p in paths:
            j = read(p)
            jobs.setdefault(j['system'], {})[j['seed']] = j
        jobs = {s: seeds for s, seeds in jobs.items() if set(seeds) == set(SEEDS)}
        for subset in ('all', 'held-out'):
            selected = {s: ss for s, ss in jobs.items() if subset == 'all' or held(s)}
            for scorer in SC if strict_u else ('strict',):
                for n in (100, 200):
                    values = {m: {} for m in methods}
                    for s, ss in selected.items():
                        for m in methods:
                            vals = []
                            for seed in SEEDS:
                                found = [r['auc'] for r in ss[seed]['rows'] if r['method'] == m and r['ref'] == n
                                         and r.get('scorer', 'strict') == scorer]
                                if not found and strict_u and scorer == 'strict' and m == 'U-best' and kind == 'disjoint' and n == 100:
                                    original = E / f'results_phase3{MODELS[model]}' / 'unseen'
                                    chosen = read(original / 'tune_u/chosen.json')['best']
                                    old = read(original / 'run' / kind / s / f'seed{seed}.json')
                                    found = [r['auc'] for r in old['rows'] if r['method'] == 'U:'+chosen and r['ref'] == n]
                                vals.append(found[0] if found else None)
                            if any(v is not None for v in vals):
                                values[m][s] = vals
                    result = stats(values, target)
                    if result:
                        label = f'{model}/{kind}/{subset}/{scorer}/@{n}'
                        all_results[label] = result
                        table(detail, label, result, target)
    return all_results


def main():
    short = ['# CPU round 3: reviewer robustness checks', '',
             'CPU only: stored outputs, hidden states and stream caches; no new LLM calls or model/tokenizer downloads. '
             'Protocols and deviations are in PROGRESS Decisions. Seeds 0–4; mean ± sample std of seed macro means. '
             'Held-out excludes HDFS, Spark and Linux. Llama base uses 11 systems (HDFS/OpenSSH caches missing); strong Llama 13, Gemma 12.', '',
             '## R1. Does scoring convention explain the conclusions?', '']
    detail = ['# CPU round 3 details', '',
              'Scorers fixed before results: strict; L1 = official AdaParser @232951a prediction post-processing then exact GT match; '
              'L2 = L1 plus symmetric whitespace collapse and merging consecutive <*> tokens. None remains failure. '
              'L1 includes semantic abstraction of numbers/addresses/paths, beyond formatting; it can break strict matches. '
              'Ground truth is not passed through AdaParser. Label samples 5–9 and output-entropy samples 0–4 are disjoint. '
              'Original systems and strict split skips remain fixed; scorer-induced class degeneracy is reported as undefined. '
              'Every rerun asserts strict stored Clotho/Drain AUC or strict stream PA curves and N5 PA/FTA first.', '',
              'The existing U reference budget @100 contains 90 target-suite labels plus 10 synthetic initial tests; '
              'U drops synthetic rows from feature training. We retain the original 100-label budget convention for '
              'the requested equal-effort reference. U-zero uses no target-suite labels; the strong parser still '
              'has its ten verified seed templates, supplied equally to all review methods as in D. '
              'Deterministic orders repeat across seeds: five positive seeds are descriptive, not five independent datasets.', '',
              'Corrected time assumption: machine/user setup 19:01 EST is 00:01 UTC, not 19:01 UTC. '
              'Conservative deadline 10:00 UTC, long-job cutoff 09:15 UTC; actual VCL reservation end is not exposed locally.', '']
    summary = {'R1': {}, 'R2': {}, 'R3': {}}
    stream_short = []
    for model, suffix in MODELS.items():
        root = E / f'results_phase3{suffix}'
        audit = read(root / 'robustness/audit.json')['systems']
        detail += ['', '## '+model+' per-system scorer audit', '',
                   '| suite | strict/L1/L2 PASS | forgiven inputs L1/L2 | broken inputs L1/L2 | forgiven samples L1/L2 | broken samples L1/L2 | outside 10–90% | between share (40 samples) strict/L1/L2 |',
                   '|---|---|---|---|---|---|---|---|']
        for ds, r in audit.items():
            c = r['scorers']
            rates = '/'.join(f"{c[s]['pass_rate']:.3f}" for s in SC)
            counts = ['/'.join(str(c[s][key]) for s in SC[1:]) for key in
                      ('forgiven_inputs', 'broken_inputs', 'forgiven_samples', 'broken_samples')]
            var = '/'.join(f"{c[s]['reliable_between_share40']:.3f}" if c[s].get('reliable_between_share40') is not None else 'n/a' for s in SC)
            flags = ','.join(s for s in SC if c[s]['out_of_range']) or 'none'
            detail.append(f'| {ds} | {rates} | '+ ' | '.join(counts) + f' | {flags} | {var} |')
        for s in SC:
            vv = [r['scorers'][s]['reliable_between_share40'] for r in audit.values()
                  if r['scorers'][s].get('reliable_between_share40') is not None]
            if vv:
                detail += ['', f'{model} {s}: reliable between-template variance share median {np.median(vv):.3f}, '
                           f'range {min(vv):.3f}–{max(vv):.3f} ({len(vv)} capped systems).']
        summary['R1'][model] = {'suites': suite_stats(root / 'robustness/suites', model, detail,
            ('clotho', 'MDSA', 'lookup_drain', 'out_entropy', 'random', 'U-best'), strict_u=True), 'streams': {}}
        cells = completed_cells(root / 'robustness/streams')
        for parser, systems in cells.items():
            expected = 11 if model == 'llama' and parser == 'base' else 13 if model == 'llama' else 12
            if len(systems) != expected:
                detail += ['', f'R1 {model}/{parser}: {len(systems)}/{expected} completed stream systems. Incomplete coverage.']
            if not systems:
                continue
            for subset in ('all', 'held-out'):
                selected = {s: r for s, r in systems.items() if subset == 'all' or held(s)}
                for sc in SC:
                    values_by_n = {}
                    for n in (5, 10, 20, 50):
                        values = {m: {} for m in ('random', 'frequent', 'dsize', 'oracle')}
                        for s, r in selected.items():
                            for m in values:
                                values[m][s] = [r['scorers'][sc]['runs'][m][str(k)]['share'][str(n)] for k in SEEDS]
                        st = stats(values, 'frequent')
                        values_by_n[n] = st
                        table(detail, f'R1 {model}/{parser}/{subset}/{sc} share fixed @{n}', st, 'frequent')
                    cells_sc = [r['scorers'][sc] for r in selected.values()]
                    pa0 = float(np.mean([r['pa0'] for r in cells_sc]))
                    wrong = float(np.mean([r['wrong_share'] for r in cells_sc]))
                    label = f'{parser}/{subset}/{sc}'
                    summary['R1'][model]['streams'][label] = {'pa0': pa0, 'wrong_share': wrong,
                        'n_systems': len(selected), 'review': values_by_n}
                    detail += ['', f'R1 {model}/{label}: PA_0 {pa0:.3f}, wrong-group share {wrong:.3f}.', '',
                               '| system | PA_0 | wrong groups | size/wrong Spearman | top20 wrong | Drain PA/FTA/rare/frequent | LLM PA/FTA/rare/frequent/excl seeds |',
                               '|---|---|---|---|---|---|---|']
                    for s, r in selected.items():
                        c = r['scorers'][sc]
                        dt, ll = c['n5']['Drain'], c['n5'][parser]
                        f = lambda x, keys: '/'.join('n/a' if x[k] is None else f'{x[k]:.3f}' for k in keys)
                        detail.append(f"| {s} | {c['pa0']:.3f} | {c['wrong_share']:.3f} | {c['size_wrong_rho']} | {c['top20_wrong']:.3f} | "
                                      + f(dt, ('PA', 'FTA', 'rare', 'frequent'))+' | '+f(ll, ('PA', 'FTA', 'rare', 'frequent', 'exclude_seeds'))+' |')
                    if subset == 'all':
                        st20 = values_by_n[20]
                        ratio = np.mean(st20['frequent']['per_seed'])/np.mean(st20['oracle']['per_seed']) if st20 else float('nan')
                        stream_short.append((model, parser, sc, len(selected), pa0, ratio, st20.get('dsize', {})))
        # N1 proxy explanation across all complete cells, separate scorer and N.
        from scipy.stats import pearsonr
        for sc in SC:
            for n in (5, 10, 20):
                actual, proxy = [], []
                for parser, systems in cells.items():
                    for r in systems.values():
                        c = r['scorers'][sc]
                        if c['proxy'][str(n)]['oracle'] is None:
                            continue
                        a = c['runs']['oracle']['0']['share'][str(n)]
                        b = c['runs']['frequent']['0']['share'][str(n)]
                        if a is None or b is None:
                            continue
                        actual.append(a-b)
                        proxy.append(c['proxy'][str(n)]['oracle']-c['proxy'][str(n)]['frequent'])
                if len(actual) > 2:
                    detail += ['', f'N1 {model} {sc} @{n}: proxy gap vs actual r={pearsonr(actual, proxy).statistic:.3f}, '
                               f'MAE={np.mean(np.abs(np.array(actual)-proxy)):.4f}, {len(actual)} cell/system points.']
    if stream_short:
        wins = [(m, p, s) for m, p, s, _, _, _, d in stream_short if d.get('positive_seeds') == 5]
        short += [('**The 20-review frequency claim changes under lenient scoring:** D x size beats Most frequent on all five seeds for '
                   +', '.join('/'.join(x) for x in wins)+'.') if wins else '**The 20-review frequency claim survives the completed stream cells:** D x size does not beat Most frequent on all five seeds in any model/parser/scorer cell.', '',
                  '“Close to Oracle” is quantified below, rather than assumed. L1/L2 are sensitivity analyses, not a new ground truth.', '',
                  '| model/parser | scorer | systems | PA_0 | Most frequent / Oracle @20 | D minus frequency @20 (positive seeds) |',
                  '|---|---|---:|---:|---:|---|']
        for m, p, s, count, pa, ratio, d in stream_short:
            short.append(f"| {m}/{p} | {s} | {count} | {pa:.3f} | {ratio:.3f} | {d.get('delta', 'n/a')} ({d.get('positive_seeds', 0)}/5) |")
    else:
        short += ['R1 stream results are still running; no complete-model headline yet.']
    short += ['', 'Suite PASS rates increase under L1/L2. The disjoint U-best advantage over Clotho remains; full AUC, variance-share, N1 and N5 results are in the details.', '']
    # R2 and R3 can be appended independently as their complete cell files arrive.
    for number, folder in [(2, 'robustness_tail'), (3, 'zero_target')]:
        short += [f'## R{number}. '+('Template metrics and the tail' if number == 2 else 'U without target-system labels'), '']
        headline_rows, wins = [], []
        for model, suffix in MODELS.items():
            root = E / f'results_phase3{suffix}'
            if number == 3:
                summary['R3'][model] = {'suites': suite_stats(root / folder / 'suites', model, detail,
                    ('clotho', 'MDSA', 'lookup_drain', 'out_entropy', 'random', 'U-zero', 'U+V5-zero', 'D-zero'))}
            cells = completed_cells(root / folder if number == 2 else root / folder / 'streams')
            for parser, ss in cells.items():
                if not ss:
                    continue
                for subset in ('all', 'held-out'):
                    systems = {s: r for s, r in ss.items() if subset == 'all' or held(s)}
                    for scope in ('full', 'tail') if number == 2 else ('review',):
                        budgets = (5, 10, 20, 50, 100) if scope == 'full' else (5, 10, 20, 50)
                        for n in budgets:
                            for metric in ('groups', 'FTA', 'lines'):
                                values = {}
                                oracle_name = 'Oracle-'+metric
                                if number == 2 and any(not r[scope]['oracle'][str(n)][metric]['certified'] for r in systems.values()):
                                    oracle_name += ' (uncertified incumbent)'
                                for s, r in systems.items():
                                    by = r[scope]['runs'] if number == 2 else r['review']
                                    for method, runs in by.items():
                                        values.setdefault(method, {})[s] = [runs[str(k)]['metrics'][str(n)][metric] for k in SEEDS]
                                    if number == 2:
                                        val = r[scope]['oracle'][str(n)][metric]['value']
                                        values.setdefault(oracle_name, {})[s] = [val]*5
                                    else:
                                        other = root / 'robustness_tail' / parser / f'{s}.json'
                                        if other.exists() and read(other).get('complete'):
                                            other = read(other)['full']['runs']
                                            for method in ('frequent', 'dsize'):
                                                values.setdefault(method, {})[s] = [other[method][str(k)]['metrics'][str(n)][metric] for k in SEEDS]
                                st = stats(values, 'frequent')
                                label = f'{model}/{parser}/{subset}/{scope}/{metric}/@{n}'
                                summary[f'R{number}'][label] = st
                                if st:
                                    table(detail, f'R{number} '+label, st, 'frequent')
                                if n == 20 and subset == 'all' and st:
                                    candidates = [m for m in st if m not in ('frequent', 'random') and not m.startswith('Oracle')
                                                  and not m.startswith('entropy')]
                                    best = max(candidates, key=lambda m: st[m]['delta_mean']) if candidates else 'frequent'
                                    headline_rows.append((model, parser, scope, metric, best, st[best]))
                                    if st[best]['positive_seeds'] == 5:
                                        wins.append(f'{model}/{parser}/{scope}/{metric}: {best} {st[best]["delta"]}')
                if number == 2:
                    if all('switch' in r for r in ss.values()):
                        for n in (5, 10, 20, 50):
                            for metric in ('groups', 'FTA', 'lines'):
                                values = {'frequent': {}}
                                for s, r in ss.items():
                                    values['frequent'][s] = [r['switch']['frequent'][str(n)][metric]]*5
                                    for name, seeds in r['switch']['runs'].items():
                                        values.setdefault(name, {})[s] = [seeds[str(k)]['metrics'][str(n)][metric] for k in SEEDS]
                                st = stats(values, 'frequent')
                                label = f'{model}/{parser}/switch20+{n}/{metric}'
                                summary['R2'][label] = st
                                if st:
                                    table(detail, label+' FULL-stream outcome at matched total budget', st, 'frequent')
                    # Oracle certification and equal-effort comparison, never hide a bound.
                    uncert = sum(not x['certified'] for r in ss.values() for scope in ('full', 'tail')
                                 for by in r[scope]['oracle'].values() for x in by.values())
                    detail += ['', f'R2 {model}/{parser}: {uncert} oracle solutions not certified optimal (inspect job solver status/gap).', '',
                               '| equal-effort reference @20 | U+V5 (100 target labels) | Most frequent @100 (capped; requested N+100=120) |', '|---|---|---|']
                    for n in (5, 10, 20, 50):
                        for metric in ('groups', 'FTA', 'lines'):
                            u, f = [], []
                            for k in SEEDS:
                                u.append(np.nanmean([float('nan') if r['full']['runs']['usize'][str(k)]['metrics'][str(n)][metric] is None else r['full']['runs']['usize'][str(k)]['metrics'][str(n)][metric] for r in ss.values()]))
                                f.append(np.nanmean([float('nan') if r['full']['runs']['frequent'][str(k)]['metrics']['100'][metric] is None else r['full']['runs']['frequent'][str(k)]['metrics']['100'][metric] for r in ss.values()]))
                            detail.append(f'| {metric}, N={n} | {fmt(u)} | {fmt(f)} |')
                else:
                    for name in ('U-zero', 'U+V5-zero'):
                        for q in ('0.05', '0.1', '0.2'):
                            values = {name: {}, 'D gate': {}}
                            for s, r in ss.items():
                                values[name][s] = [r['gate'][name][q][str(k)]['share'] for k in SEEDS]
                                chosen = read(root / 'gate/tune/chosen.json')['w']
                                pp = [root / 'gate/run' / parser / s / f'dgate_w{chosen}_q{round(float(q)*100)}_s{k}.json' for k in SEEDS]
                                if all(p.exists() for p in pp):
                                    values['D gate'][s] = [read(p)['share'] for p in pp]
                            st = stats(values, 'D gate')
                            if st:
                                label = f'{model}/{parser}/gate/{name}/q{q}'
                                summary['R3'][label] = st
                                table(detail, label+' share of bad lines fixed', st, 'D gate')
                            detail += ['', f'R3 {model}/{parser}/{name}/q{q}: gate metrics', '',
                                       '| metric | mean ± std |', '|---|---|']
                            for metric in ('N', 'precision', 'share'):
                                per = [float(np.nanmean([r['gate'][name][q][str(k)][metric] for r in ss.values()])) for k in SEEDS]
                                detail.append(f'| {metric} | {fmt(per)} |')
        if headline_rows:
            short += [('**At 20 reviews, gains over frequency on every seed occur in:** '+ '; '.join(wins)+'.') if wins else
                      '**At 20 reviews, none of the evaluated zero-extra-call predictors beats frequency on every seed in the completed cells.**', '',
                      'U in R2 uses 100 target labels; U-zero in R3 uses none. Output entropy requires extra samples and is excluded from this headline.', '',
                      'Only completed systems are shown; the system count below makes partial coverage explicit. '
                      'R2 details also compare frequency-20 then risk with uninterrupted frequency at equal total budgets.', '',
                      '| model/parser | systems | scope | metric | largest observed predictor gain | minus frequency (positive seeds) |',
                      '|---|---:|---|---|---|---|']
            for m, p, scope, metric, best, r in headline_rows:
                short.append(f"| {m}/{p} | {len(r['systems'])} | {scope} | {metric} | {best} | {r['delta']} ({r['positive_seeds']}/5) |")
        else:
            short += ['Pending: no complete cell available.']
        short += ['']
    short += ['## R4. Full-length concentration', '']
    full = {p.stem: read(p) for p in (E / 'results_phase3/concentration_full').glob('*.json') if read(p).get('complete')}
    summary['R4'] = full
    if full:
        short += [f'Concentration measured on {len(full)} systems beyond the original stretch. **Wrongness is not measured here.** '
                  'Existing ZIPs are streamed one system at a time, with a 2M-line cap and no raw extraction. '
                  'These counts are deterministic; seed standard deviations are zero.', '',
                  '| checkpoint | systems | actual lines | mean groups | mean Gini | mean top1/5/10/20 share | mean new-after-50k share |',
                  '|---|---:|---|---:|---:|---|---:|']
        for k in (50_000, 200_000, 1_000_000, 'full'):
            points = [r['points'][str(max(map(int, r['points'])))] if k == 'full' else r['points'].get(str(k)) for r in full.values()]
            points = [p for p in points if p]
            if not points:
                continue
            means = lambda key: float(np.mean([p[key] for p in points]))
            tops = '/'.join(f"{np.mean([p['top_share'][str(v)] for p in points]):.3f}" for v in (1, 5, 10, 20))
            low, high = min(p['n_lines'] for p in points), max(p['n_lines'] for p in points)
            label = '2M/full' if k == 'full' else f'{k:,}'
            short.append(f"| {label} | {len(points)} | {low:,}–{high:,} | {means('groups'):.1f} | {means('gini'):.3f} | {tops} | {means('new_after_50k_share'):.3f} |")
        detail += ['', '## R4 full-length concentration (no wrongness measured)', '',
                   '| system | lines | groups | Gini | top1/5/10/20 shares | new-after-50k share |', '|---|---:|---:|---:|---|---:|']
        for s, r in full.items():
            for n, p in r['points'].items():
                tops = '/'.join(f"{p['top_share'][str(v)]:.3f}" for v in (1, 5, 10, 20))
                detail.append(f"| {s} | {n} | {p['groups']} | {p['gini']:.3f} | {tops} | {p['new_after_50k_share']:.3f} |")
    else:
        short += ['Queued only if R1–R3 finish with at least 30 minutes before the CPU cutoff. No wrongness is measured by R4.']
    short += ['', 'All per-system artifacts and statistics are in [cpu_round3_details.md](cpu_round3_details.md). '
              'Tail budgets are additional reviews after the 20 largest eligible groups; FTA uses only remaining lines. '
              '“Largest observed gain” describes results, not a held-out method-selection rule.']
    REP.mkdir(exist_ok=True)
    (REP / 'cpu_round3.md').write_text('\n'.join(short)+'\n')
    (REP / 'cpu_round3_details.md').write_text('\n'.join(detail)+'\n')
    (REP / 'cpu_round3_details.json').write_text(json.dumps(summary, indent=1))
    print('Updated round 3 report from complete artifacts')


if __name__ == '__main__':
    main()

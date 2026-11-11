E0–E2 use Llama capped suites, rules_template, strict match, seed 0 and reference size 100 where applicable; pass rates are fractions of suite inputs labelled PASS (≥3 correct among outputs 6–10), not generation accuracy.

| Section | Stored sources / decisions |
|---|---|
| Setup / E0 | data/results_llama/log_parsing/inference_results/rules_template/; suite JSONL |
| Where do failures happen / E1 | data/template_difficulty/{llama,gemma}.json; reports/template_difficulty.json |
| Can we predict failures / E2 | results_phase3/final/fair_leakfree/{standard,disjoint}/; data/unseen_feats/llama/ |
| Does prediction help human review / E3 | data/stream/; data/stream_gen/gemma/; results_phase3_gemma/triage/e1/base/ |
| Other checks / E4 | Same BGL caches; N3 words = [a-z]+ on lowercase text after replacing <*> with a space |
| Time / scope | Deadline supplied by user: 2026-11-11 18:15 EST; long-job cutoff 17:30, persist by 17:45; stored-file extraction only, no predictors fitted or LLM calls |

E0 rule: systems alphabetically, then suite index ascending; first mixed 10-output line ≤160 characters; first correct and first wrong parsed outputs in sample order (Apache index 0; skipped 0 overlong candidates).

| System / index | Log line | Ground truth | Correct parsed output | Wrong parsed output | Outputs 6–10 | Label |
|---|---|---|---|---|---|---|
| Apache / 0 | `[client 218.94.1.192] Directory index forbidden by rule: /var/www/html/` | `[client <*>] Directory index forbidden by rule: <*>` | `[client <*>] Directory index forbidden by rule: <*>` | `[client <*>] Directory <*> by <*>: /<*>/<*>` | PASS FAIL FAIL PASS FAIL | FAIL |

E1 Easy 1 rule: top two by suite-line count among templates with PASS rate 1 for both models and only space-separated <*> tokens; ties system/template; first suite index ≤160 characters, index 9 (skipped 0 overlong lines).

| Factor / system | Real line | Ground-truth template | Most common wrong (all 10 outputs, all template lines; lexical ties) | Llama PASS | Gemma PASS | Suite lines |
|---|---|---|---|---|---|---|
| Easy 1 / OpenSSH | `Connection closed by 58.42.244.194 [preauth]` | `Connection closed by <*> [preauth]` | — | 1.000 | 1.000 | 28 |

E1 Easy 2 rule: top two by suite-line count among templates with PASS rate 1 for both models and only space-separated <*> tokens; ties system/template; first suite index ≤160 characters, index 91 (skipped 0 overlong lines).

| Factor / system | Real line | Ground-truth template | Most common wrong (all 10 outputs, all template lines; lexical ties) | Llama PASS | Gemma PASS | Suite lines |
|---|---|---|---|---|---|---|
| Easy 2 / HPC | `NIFF: node node-69 detected a failed network connection on network 5.5.0.0 via interface alt0` | `NIFF: node <*> detected a failed network connection on network <*> via interface <*>` | — | 1.000 | 1.000 | 16 |

E1 Attached rule: first illustrative template in the named factor in template_difficulty.md; first suite index ≤160 characters, index 3 (skipped 0 overlong lines).

| Factor / system | Real line | Ground-truth template | Most common wrong (all 10 outputs, all template lines; lexical ties) | Llama PASS | Gemma PASS | Suite lines |
|---|---|---|---|---|---|---|
| Attached / OpenSSH | `PAM 5 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=219.151.22.59 user=root` | `PAM <*> more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>` | `PAM 5 more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>` (68 occurrences) | 0.000 | 0.000 | 28 |

E1 Word-like rule: first illustrative template in the named factor in template_difficulty.md; first suite index ≤160 characters, index 3 (skipped 0 overlong lines).

| Factor / system | Real line | Ground-truth template | Most common wrong (all 10 outputs, all template lines; lexical ties) | Llama PASS | Gemma PASS | Suite lines |
|---|---|---|---|---|---|---|
| Word-like / OpenSSH | `PAM 5 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=219.151.22.59 user=root` | `PAM <*> more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>` | `PAM 5 more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>` (68 occurrences) | 0.000 | 0.000 | 28 |

E1 Path rule: first illustrative template in the named factor in template_difficulty.md; first suite index ≤160 characters, index 7 (skipped 0 overlong lines).

| Factor / system | Real line | Ground-truth template | Most common wrong (all 10 outputs, all template lines; lexical ties) | Llama PASS | Gemma PASS | Suite lines |
|---|---|---|---|---|---|---|
| Path / OpenSSH | `Bad protocol version identification 'GET /manager/html HTTP/1.1' from 183.131.83.223 port 59370` | `Bad protocol version identification <*> from <*> port <*>` | `Bad protocol version identification '<*>' from <*> port <*>` (90 occurrences) | 0.000 | 0.000 | 22 |

E2 standard rule: median seed-0 standard Clotho AUC @100 among 13 systems, tie by system → Zookeeper (0.752430); lexicographically first (FAIL reference index, same-template test index) with both lines ≤160 characters; references = complement of stored test indices.

| Role / index | Line | Stored ground-truth template | Label | Clotho risk percentile |
|---|---|---|---|---|
| Reference / 8 | `Notification: 1 (n.leader), 0x8000000bc (n.zxid), 0x1 (n.round), LOOKING (n.state), 1 (n.sid), 0x8 (n.peerEPoch), LOOKING (my state)` | `Notification: <*> (n.leader), <*> (n.zxid), <*> (n.round), <*> (n.state), <*> (n.sid), <*> (n.peerEPoch), <*> (my state)` | FAIL | — |
| Test / 49 | `Notification: 3 (n.leader), 0x300000dcd (n.zxid), 0x5 (n.round), LOOKING (n.state), 3 (n.sid), 0x4 (n.peerEPoch), LOOKING (my state)` | `Notification: <*> (n.leader), <*> (n.zxid), <*> (n.round), <*> (n.state), <*> (n.sid), <*> (n.peerEPoch), <*> (my state)` | FAIL | Unavailable: per-input scores not saved |

E2 disjoint rule: among FAIL test inputs choose lowest Clotho risk, tie by suite index, then skip lines >160 characters; unavailable because no stored per-input Clotho ranks exist (aggregate AUC cannot select this example).

| Requested example | Line / template | Clotho percentile | U-best percentile | Output-entropy percentile |
|---|---|---|---|---|
| Lowest-risk disjoint FAIL | Not selected; no predictor rerun permitted | Not stored | Not stored | oe exists in feature cache, but no selected input to rank |

E3 rule: Gemma base, lower median of 12 systems sorted by Most frequent strict mis-parsed-line share fixed @20, tie by system → OpenStack (0.819699); groups by descending size then group id; correctness means cached template equals majority GT (mixed groups still contain residual errors).

| System | Groups | Lines | PA_0 |
|---|---|---|---|
| OpenStack | 2616 | 50000 | 0.149200 |

E3 largest-groups rule: first 10 of descending group size, tie by group id; majority GT ties use first seen.

| Group | Lines | Cached template | Majority ground-truth template | Verdict |
|---|---|---|---|---|
| 0 | 23330 | `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>` | `<*> "GET <*> <*>" status: <*> len: <*> time: <*>` | wrong |
| 1297 | 2173 | `[instance: <*>] VM Started (Lifecycle Event)` | `[instance: <*>] VM Resumed (Lifecycle Event)` | wrong |
| 6 | 2066 | `image <*> at (<*>) : checking` | `image <*> at (<*>): checking` | wrong |
| 7 | 2066 | `image <*> at (<*>): in use: on this node <*>, <*> on other nodes sharing this instance storage` | `image <*> at (<*>): in use: on this node <*> local, <*> on other nodes sharing this instance storage` | wrong |
| 8 | 2065 | `Active base files: /var/lib/nova/instances/_base/<*>` | `Active base files: <*>` | wrong |
| 10 | 1596 | `<*> "POST <*> HTTP/1.1" status: <*> len: <*> time: <*>` | `<*> "POST <*> <*>" status: <*> len: <*> time: <*>` | wrong |
| 1311 | 867 | `[instance: <*>] Instance spawned successfully.` | `[instance: <*>] Instance spawned successfully.` | correct |
| 1312 | 867 | `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.` | `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.` | correct |
| 1319 | 864 | `[instance: <*>] disk limit not specified, defaulting to unlimited` | `[instance: <*>] disk limit not specified, defaulting to unlimited` | correct |
| 1310 | 849 | `[instance: <*>] During sync_power_state the instance has a pending task (<*>). Skip.` | `[instance: <*>] During sync_power_state the instance has a pending task (spawning). Skip.` | wrong |

E3 tail rule: remove the 20 largest groups, then take the first 3 wrong groups in descending size / group id order.

| Group | Lines | Cached template | Majority ground-truth template | Verdict |
|---|---|---|---|---|
| 1316 | 432 | `[instance: <*>] Attempting claim: memory <*> , disk <*> , vcpus <*> CPU` | `[instance: <*>] Attempting claim: memory <*> MB, disk <*> GB, vcpus <*> CPU` | wrong |
| 27 | 364 | `Auditing locally available compute resources for node <*>.` | `Auditing locally available compute resources for node <*>` | wrong |
| 28 | 364 | `Total usable <*>, total allocated <*>` | `Total usable vcpus: <*>, total allocated vcpus: <*>` | wrong |

E3 review rule: replay each stored seed-0 sequence, first 5 groups; fixed lines = majority-GT correction gain, with total gain after 5 (no new ordering computed).

| Order | Review 1: group: size / fixed | Review 2 | Review 3 | Review 4 | Review 5 | Total fixed after 5 |
|---|---|---|---|---|---|---|
| Oracle | 0: 23330 / 23330 | 6: 2066 / 2066 | 7: 2066 / 2066 | 8: 2065 / 2065 | 10: 1596 / 1596 | 31123 |
| Most frequent | 0: 23330 / 23330 | 1297: 2173 / 435 | 6: 2066 / 2066 | 7: 2066 / 2066 | 8: 2065 / 2065 | 29962 |
| D × size | 0: 23330 / 23330 | 6: 2066 / 2066 | 7: 2066 / 2066 | 1297: 2173 / 435 | 10: 1596 / 1596 | 29493 |
| Clotho alone | 29: 364 / 364 | 63: 1 / 1 | 456: 1 / 0 | 1: 2 / 0 | 6: 2066 / 2066 | 2431 |

E4 rule: BGL Gemma base, wrong groups whose complete line-index set equals a single GT template’s line-index set; largest first, tie group id; words use the exact N3 tokenizer, preserving order and duplicates.

| Group | Lines | Cached template | Ground-truth template | Cached word list | GT word list |
|---|---|---|---|---|---|
| 2 | 37788 | `generating <*>.<*>` | `generating core.<*>` | `generating` | `generating, core` |

E4 explanation: identical line sets preserve this event’s count-vector column (up to renaming); different template words change the N3 TF-IDF representation.

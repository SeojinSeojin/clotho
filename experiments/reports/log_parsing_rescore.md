# log_parsing rescoring (no regeneration)

Input PASS = more than 5 of 10 samples match. Cells: input PASS rate / output match rate. `log_parsing_rules` = legacy results from an uncommitted rules prompt (reference only).

| task | template | system | strict_match | normalized_match | lenient_match | parse fail |
|---|---|---|---|---|---|---|
| log_parsing | messages_template | apache | 43.8% / 43.1% | 44.0% / 43.2% | 82.6% / 73.4% | 0.0% |
| log_parsing | messages_template | bgl | 33.0% / 35.1% | 33.4% / 35.7% | 58.6% / 56.7% | 0.9% |
| log_parsing | messages_template | hadoop | 21.2% / 24.0% | 21.4% / 24.0% | 71.4% / 67.4% | 0.3% |
| log_parsing | messages_template | hdfs | 2.6% / 9.9% | 2.6% / 10.6% | 44.0% / 48.8% | 0.0% |
| log_parsing | messages_template | healthapp | 44.0% / 44.0% | 46.6% / 46.8% | 73.8% / 71.3% | 0.1% |
| log_parsing | messages_template | hpc | 18.0% / 24.7% | 21.0% / 27.3% | 46.2% / 48.8% | 0.2% |
| log_parsing | messages_template | linux | 27.8% / 29.3% | 27.8% / 29.3% | 56.6% / 58.5% | 0.4% |
| log_parsing | messages_template | mac | 18.6% / 21.1% | 18.8% / 21.4% | 35.0% / 38.1% | 0.5% |
| log_parsing | messages_template | openssh | 35.6% / 38.3% | 35.8% / 38.6% | 55.6% / 56.4% | 0.0% |
| log_parsing | messages_template | openstack | 55.6% / 52.1% | 55.6% / 52.1% | 79.4% / 72.4% | 0.0% |
| log_parsing | messages_template | proxifier | 0.0% / 0.2% | 0.0% / 0.2% | 40.4% / 44.1% | 0.0% |
| log_parsing | messages_template | spark | 22.4% / 25.6% | 22.4% / 25.7% | 59.0% / 59.3% | 0.4% |
| log_parsing | messages_template | thunderbird | 30.0% / 30.6% | 30.4% / 31.4% | 53.2% / 54.1% | 0.6% |
| log_parsing | messages_template | zookeeper | 32.0% / 36.4% | 32.0% / 36.5% | 68.0% / 67.8% | 0.3% |
| log_parsing | rules_template | apache | 57.0% / 55.1% | 57.0% / 55.1% | 80.6% / 71.6% | 0.0% |
| log_parsing | rules_template | bgl | 20.0% / 27.0% | 20.2% / 27.3% | 42.2% / 44.4% | 0.6% |
| log_parsing | rules_template | hadoop | 28.6% / 31.8% | 28.6% / 31.8% | 58.6% / 59.7% | 0.2% |
| log_parsing | rules_template | hdfs | 27.2% / 30.4% | 27.2% / 30.6% | 43.2% / 45.1% | 0.0% |
| log_parsing | rules_template | healthapp | 29.6% / 34.5% | 31.6% / 36.7% | 63.0% / 63.0% | 0.0% |
| log_parsing | rules_template | hpc | 18.4% / 26.8% | 21.2% / 29.5% | 40.4% / 44.6% | 0.0% |
| log_parsing | rules_template | linux | 22.8% / 26.8% | 22.8% / 26.9% | 38.2% / 46.0% | 0.9% |
| log_parsing | rules_template | mac | 12.0% / 17.1% | 12.0% / 17.3% | 24.2% / 29.5% | 0.4% |
| log_parsing | rules_template | openssh | 35.4% / 41.0% | 36.0% / 41.5% | 46.8% / 50.3% | 0.0% |
| log_parsing | rules_template | openstack | 54.2% / 51.9% | 54.2% / 51.9% | 69.8% / 64.5% | 0.0% |
| log_parsing | rules_template | proxifier | 0.0% / 0.7% | 0.0% / 0.7% | 31.4% / 39.2% | 0.0% |
| log_parsing | rules_template | spark | 22.0% / 26.9% | 22.2% / 27.0% | 49.2% / 51.2% | 0.2% |
| log_parsing | rules_template | thunderbird | 20.4% / 24.7% | 20.6% / 25.0% | 38.4% / 42.2% | 0.4% |
| log_parsing | rules_template | zookeeper | 44.6% / 44.8% | 44.8% / 44.8% | 60.2% / 63.1% | 0.3% |
| log_parsing_rules | messages_template | hdfs | 37.8% / 39.5% | 38.0% / 40.8% | 55.2% / 59.0% | 0.0% |

# log_parsing rescoring (no regeneration)

Input PASS = more than 5 of 10 samples match. Cells: input PASS rate / output match rate. `log_parsing_rules` = legacy results from an uncommitted rules prompt (reference only).

| task | template | system | strict_match | normalized_match | lenient_match | parse fail |
|---|---|---|---|---|---|---|
| log_parsing | messages_template | hdfs | 2.6% / 9.9% | 2.6% / 10.6% | 44.0% / 48.8% | 0.0% |
| log_parsing_rules | messages_template | hdfs | 37.8% / 39.5% | 38.0% / 40.8% | 55.2% / 59.0% | 0.0% |

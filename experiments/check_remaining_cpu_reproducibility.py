"""Repeat new B/C computations only; previous completed experiments remain untouched."""
from cpu_research_common import *
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor
ROOT=EXP/'results_phase3/cpu_remaining'
def main():
 names=['correction_opportunity.json','correction_opportunity.csv','correction_mixed_groups.json.gz','correction_sources.json','calibration_metrics.json','calibration_predictions.json.gz','calibration_manifests.json','calibration_score_audit.json','calibration_fits.json','calibration_paired.json','calibration_validation.json','calibration_unavailable.json','calibration_per_system.csv']
 before={n:sha(ROOT/n) for n in names}
 def run(script):
  log=ROOT/'repro_logs'/f'{script}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:subprocess.run([sys.executable,str(EXP/script)],check=True,stdout=f,stderr=subprocess.STDOUT)
 with ThreadPoolExecutor(2) as pool:list(pool.map(run,['log_parsing_correction_opportunity.py','log_parsing_predictor_calibration.py']))
 after={n:sha(ROOT/n) for n in names};different=[n for n in names if before[n]!=after[n]]
 save(ROOT/'reproducibility.json',{'files':len(names),'all_identical':not different,'differing_files':different,'before_sha256':before,'after_sha256':after,'PDF_creation_metadata_excluded':True,'full_raw_scans_not_repeated':'all-row join assertions, original archive hashes and independently checked manifests preserve provenance','ko':'새로운 교정 상한/보정 분석만 반복해13개 수치 결과 및 확률 파일의 바이트 해시가 같은지 확인했다. 기존 완료 실험이나16.6M줄 원본 스캔은 재실행하지 않았다.'});assert not different,different;print('identical numeric/probability artifacts',len(names))
if __name__=='__main__':main()

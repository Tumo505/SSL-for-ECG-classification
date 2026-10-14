"""Download and validate MIT-BIH before installing the complete data folder."""
from pathlib import Path
import shutil
import wfdb

root = Path(__file__).resolve().parents[1]
target = root/'Data/mit-bih-arrhythmia-database-1.0.0'
staging = Path('/private/tmp/ecg_mitbih_download')
if target.exists():
    raise SystemExit('MIT-BIH data folder already exists; refusing to replace it')
wfdb.dl_database('mitdb', str(staging), annotators=['atr'])
records = wfdb.get_record_list('mitdb')
for record in records:
    waveform = wfdb.rdrecord(str(staging/record))
    annotation = wfdb.rdann(str(staging/record), 'atr')
    import numpy as np
    if waveform.fs != 360 or not np.isfinite(waveform.p_signal).all() or not len(annotation.sample):
        raise ValueError(f'Invalid record {record}')
shutil.move(str(staging), str(target))
print(f'Validated {len(records)} MIT-BIH records: {target}')

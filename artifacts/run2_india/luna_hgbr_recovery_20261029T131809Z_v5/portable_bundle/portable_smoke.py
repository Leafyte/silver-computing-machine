import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from inference import load_bundle, predict_minutes
bundle=load_bundle('.')
with np.load('portable_smoke_fixture.npz',allow_pickle=False) as x, np.load('portable_smoke_expected.npz',allow_pickle=False) as e:
    for mode in ('operational_release','candidate_hybrid','candidate_selective'):
        got=predict_minutes(bundle,x['gbdt_features'],x['q_features'],mode=mode)['eta_minutes']
        assert np.allclose(got,e[mode],rtol=0,atol=1e-12),mode
print('portable inference smoke passed')

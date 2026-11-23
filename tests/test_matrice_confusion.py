import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "aggregation"))
import matrice_confusion as mc  # noqa: E402


def test_matrice_et_metriques():
    cp = pd.DataFrame({"classe": ["itc", "itc", "micro", "macro"], "classe_pred": ["itc", "micro", "micro", "macro"]})
    m = mc.matrice(cp)
    assert m.loc["itc", "itc"] == 1 and m.loc["itc", "micro"] == 1 and int(m.values.sum()) == 4
    rp = mc.rappel_precision(m).set_index("classe")
    assert rp.loc["itc", "rappel"] == 0.5 and rp.loc["micro", "precision"] == 0.5 and rp.loc["macro", "rappel"] == 1.0

import sys, zipfile
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src" / "aggregation"))
from pn_rules import stade_pn

def test_cas_limites():
    assert stade_pn(["negative"] * 5) == "pN0"
    assert stade_pn(["itc", "itc"] + ["negative"] * 3) == "pN0(i+)"
    assert stade_pn(["micro"] + ["negative"] * 4) == "pN1mi"
    assert stade_pn(["micro", "itc", "negative", "negative", "negative"]) == "pN1mi"
    assert stade_pn(["macro"] + ["negative"] * 4) == "pN1"
    assert stade_pn(["macro", "micro", "micro"] + ["negative"] * 2) == "pN1"
    assert stade_pn(["macro", "micro", "micro", "micro"] + ["negative"]) == "pN2"
    assert stade_pn(["macro", "itc", "itc", "itc", "itc"]) == "pN1"

def test_100_patientes_officielles():
    st = pd.read_csv(ROOT / "donnees" / "stages.csv")
    gl = st[st.patient.str.endswith(".tif")].assign(pid=lambda d: d.patient.str[:11])
    pt = st[st.patient.str.endswith(".zip")].assign(pid=lambda d: d.patient.str[:11]).set_index("pid")
    calc = gl.groupby("pid").stage.apply(stade_pn)
    assert len(pt) == 100 and (pt.stage == calc.reindex(pt.index)).all()

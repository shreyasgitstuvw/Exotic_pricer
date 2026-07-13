import sys, io, zipfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestonnn.data import bhavcopy as bc


def _new_format_df():
    return pd.DataFrame({
        "TradDt": ["2026-07-10"]*4, "FinInstrmTp": ["IDO"]*4, "TckrSymb": ["NIFTY"]*4,
        "XpryDt": ["2026-07-14","2026-07-14","2026-08-25","2026-08-25"],
        "StrkPric": [24000,24000,24000,24000], "OptnTp": ["CE","PE","CE","PE"],
        "ClsPric": [250,40,600,300], "SttlmPric": [251,41,601,301],
        "UndrlygPric": [24200]*4, "OpnIntrst": [1000]*4, "TtlTradgVol": [50]*4,
    })


def _old_format_df():
    return pd.DataFrame({
        "INSTRUMENT": ["OPTIDX"]*2, "SYMBOL": ["NIFTY"]*2, "EXPIRY_DT": ["2026-07-14"]*2,
        "STRIKE_PR": [24000,24000], "OPTION_TYP": ["CE","PE"], "CLOSE": [250,40],
        "SETTLE_PR": [251,41], "OPEN_INT": [1000,1000], "CONTRACTS": [50,50],
        "TIMESTAMP": ["2026-07-10","2026-07-10"],
    })


def test_detect_and_parse_new(tmp_path):
    p = tmp_path/"n.csv"; _new_format_df().to_csv(p, index=False)
    assert bc.detect_format(_new_format_df().columns) == "new"
    day = bc.load_bhavcopy_day(p)
    assert set(["date","expiry","tte_yr","K","right","px"]).issubset(day.columns)
    assert day["px"].iloc[0] == 251           # settlement preferred over close
    assert day["expiry"].nunique() == 2       # term structure present
    assert (day["tte_yr"] > 0).all()


def test_detect_and_parse_old(tmp_path):
    p = tmp_path/"o.csv"; _old_format_df().to_csv(p, index=False)
    assert bc.detect_format(_old_format_df().columns) == "old"
    day = bc.load_bhavcopy_day(p)
    assert len(day) == 2 and day["right"].tolist() == ["CE","PE"]


def test_zip_reading(tmp_path):
    p = tmp_path/"z.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("inner.csv", _new_format_df().to_csv(index=False))
    day = bc.load_bhavcopy_day(p)
    assert len(day) == 4

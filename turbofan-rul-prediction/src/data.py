from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SETTINGS = [f"op{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS
SENSOR_SYMBOL = {k: v.split(" - ")[0] for k, v in SENSOR_NAMES.items()}
SENSOR_NAMES = {
    "s1": "T2 - Total temperature at fan inlet",
    "s2": "T24 - Total temperature at LPC outlet",
    "s3": "T30 - Total temperature at HPC outlet",
    "s4": "T50 - Total temperature at LPT outlet",
    "s5": "P2 - Pressure at fan inlet",
    "s6": "P15 - Total pressure in bypass duct",
    "s7": "P30 - Total pressure at HPC outlet",
    "s8": "Nf - Physical fan speed",
    "s9": "Nc - Physical core speed",
    "s10": "epr - Engine pressure ratio (P50/P2)",
    "s11": "Ps30 - Static pressure at HPC outlet",
    "s12": "phi - Ratio of fuel flow to Ps30",
    "s13": "NRf - Corrected fan speed",
    "s14": "NRc - Corrected core speed",
    "s15": "BPR - Bypass ratio",
    "s16": "farB - Burner fuel-air ratio",
    "s17": "htBleed - Bleed enthalpy",
    "s18": "Nf_dmd - Demanded fan speed",
    "s19": "PCNfR_dmd - Demanded corrected fan speed",
    "s20": "W31 - HPT coolant bleed",
    "s21": "W32 - LPT coolant bleed",
}

RUL_CAP = 125  # piecewise-linear target: early life is treated as "healthy"
VALID_SPLITS = {"train", "test"}
VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}

# Internal helpers: _check_split and _check_subset
def _check_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

def _check_subset(subset: str) -> None:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(VALID_SUBSETS)}, got {subset!r}")


def load_split(split: str = "train", subset: str = "FD001") -> pd.DataFrame:
    """Load train_FDxxx.txt or test_FDxxx.txt (space-separated, no header)."""
    _check_split(split)
    _check_subset(subset)
    path = RAW_DIR / f"{split}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add RUL labels to the TRAIN set (each engine runs to failure)."""
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last_cycle - out["cycle"]
    out["rul_capped"] = out["rul"].clip(upper=cap)
    return out


def load_test_rul(subset: str = "FD001") -> pd.Series:
    """True RUL at the LAST observed cycle of each test engine, indexed by unit."""
    _check_subset(subset)
    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", header=None, names=["rul"])["rul"]
    # Row i of the RUL file belongs to the i-th engine of the test file (ascending unit ID).
    units = np.sort(load_split("test", subset)["unit"].unique())

    if len(rul) != len(units):
        raise ValueError(
            f"RUL file has {len(rul)} rows but the test file has {len(units)} engines"
        )

    rul.index = pd.Index(units, name="unit")
    return rul


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its final observed cycle (what we predict on in test)."""
    return df.loc[df.groupby("unit")["cycle"].idxmax()].set_index("unit")
"""VAZIFA 3: Kalibratsiya CSV asosida MAE va RMSE hisoblash.

CSV faqat ikkita ustun: predicted (bashorat qiymat) va actual (haqiqiy qiymat).
Fayl nomi: calibration_data.csv (skript yonida yoki --csv orqali).
"""

import argparse

import numpy as np
import pandas as pd

DEFAULT_CSV = "calibration_data.csv"


def main():
    ap = argparse.ArgumentParser(
        description="calibration_data.csv dan MAE / RMSE hisoblash")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--unit", default="", metavar="BIRLIK",
                    help="MAE/RMSE ga qo'shiladigan birlik. Misol: m3, %, tonna")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"predicted", "actual"}.issubset(df.columns):
        raise SystemExit("Xato: '{csv}' da 'predicted' va 'actual' ustunlari "
                         "boshi shart. Topilgan ustunlar: {cols}".format(
                             csv=args.csv, cols=list(df.columns)))

    df = df.dropna(subset=["predicted", "actual"])
    pred = df["predicted"].to_numpy(dtype=float)
    actual = df["actual"].to_numpy(dtype=float)

    if len(pred) == 0:
        raise SystemExit("Xato: bosh yoki son bollmagan ma'lumot.")

    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual))) * 100
    corr = float(np.corrcoef(pred, actual)[0, 1])

    u = args.unit
    print("Fayl: {}".format(args.csv))
    print("Qatorlar: {}  (nulls olib tashlandi)".format(len(df)))
    print("-" * 32)
    print("{:<15}{:<14}".format("Korsatkich", "Qiymat"))
    print("-" * 32)
    print("{:<15}{:<12.4f} {}".format("MAE", mae, u))
    print("{:<15}{:<12.4f} {}".format("RMSE", rmse, u))
    print("{:<15}{:<12.4f}%".format("MAPE", mape))
    print("{:<15}{:<12.4f}".format("Pearson r", corr))
    print("-" * 32)


if __name__ == "__main__":
    main()
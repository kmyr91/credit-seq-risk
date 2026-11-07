"""Online feature engineering matching the Spark training pipeline."""

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "mean_bill",
    "max_bill",
    "total_payment",
    "mean_repayment_status",
    "max_repayment_status",
    "months_late",
    "latest_repayment_status",
    "latest_bill",
    "bill_change",
    "credit_limit",
    "credit_utilization",
    "payment_ratio",
]


def build_online_features(statements, credit_limit):
    """Build one customer's features from six monthly statements."""

    if len(statements) != 6:
        raise ValueError("Exactly six statements are required.")

    records = [
        s.model_dump() if hasattr(s, "model_dump") else dict(s)
        for s in statements
    ]

    df = pd.DataFrame(records)
    df = df.sort_values("statement_date").reset_index(drop=True)

    bills = df["bill_amount"].to_numpy(dtype=float)
    payments = df["payment_amount"].to_numpy(dtype=float)
    statuses = df["repayment_status"].to_numpy(dtype=float)

    total_positive_bill = float(np.maximum(bills, 0).sum())
    total_payment = float(payments.sum())
    latest_bill = float(bills[-1])

    features = {
        "mean_bill": float(bills.mean()),
        "max_bill": float(bills.max()),
        "total_payment": total_payment,
        "mean_repayment_status": float(statuses.mean()),
        "max_repayment_status": float(statuses.max()),
        "months_late": int((statuses > 0).sum()),
        "latest_repayment_status": float(statuses[-1]),
        "latest_bill": latest_bill,
        "bill_change": latest_bill - float(bills[0]),
        "credit_limit": float(credit_limit),
        "credit_utilization": (
            max(latest_bill, 0.0) / float(credit_limit)
            if credit_limit != 0 else np.nan
        ),
        "payment_ratio": (
            total_payment / total_positive_bill
            if total_positive_bill != 0 else np.nan
        ),
    }

    return features

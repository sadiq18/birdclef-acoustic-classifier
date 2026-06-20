import numpy as np
import pandas as pd


def write_submission(
    row_ids: list[str],
    predictions: np.ndarray,
    primary_labels: list[str],
    output_path: str = "submission.csv",
):
    submission = pd.DataFrame(predictions, columns=primary_labels)
    submission.insert(0, "row_id", row_ids)

    assert submission["row_id"].is_unique
    assert not submission.iloc[:, 1:].isna().any().any()
    submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)

    submission.to_csv(output_path, index=False)
    print(f"Wrote {output_path}: {len(submission)} rows x {submission.shape[1]} cols")
    return submission

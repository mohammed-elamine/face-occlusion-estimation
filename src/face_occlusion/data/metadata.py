"""Path-derived metadata for diagnostics and splitting."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import pandas as pd

FACE_ID_RE = re.compile(r"FaceId-(\d+)")


def _parse_path(value: str) -> dict[str, object]:
    path = PurePosixPath(str(value))
    parts = path.parts
    database = parts[0] if parts else ""
    source_subfolder = str(path.parent) if str(path.parent) != "." else database

    # database3 folders are identity-like groups. Other databases do not expose
    # a reliable identity key, so the filename itself is the safest fallback.
    if database == "database3" and len(parts) >= 3:
        group_id = f"{database}/{parts[2]}"
    else:
        group_id = str(value)

    match = FACE_ID_RE.search(path.name)
    face_id = int(match.group(1)) if match else -1

    return {
        "database": database,
        "source_subfolder": source_subfolder,
        "group_id": group_id,
        "face_id": face_id,
    }


def add_path_metadata(df: pd.DataFrame, filename_col: str = "filename") -> pd.DataFrame:
    """Return a copy with database/source/group/face-id columns derived from paths."""

    if filename_col not in df.columns:
        raise ValueError(f"Filename column '{filename_col}' not in dataframe.")

    out = df.copy()
    parsed = out[filename_col].astype(str).map(_parse_path)
    parsed_df = pd.DataFrame(parsed.tolist(), index=out.index)
    for col in ("database", "source_subfolder", "group_id", "face_id"):
        out[col] = parsed_df[col]
    return out

"""Address/feeder search helpers -- simple substring + difflib fuzzy match.
No external fuzzy-matching dependency required (rapidfuzz not assumed
installed); difflib is stdlib and good enough at this dataset size (~8.7k
addresses, ~300 feeders)."""
from difflib import get_close_matches
import pandas as pd


def search_addresses(lookup: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    if not query or len(query.strip()) < 2:
        return lookup.head(0)
    q = query.strip().lower()
    substr_matches = lookup[lookup["ADDRESS"].str.lower().str.contains(q, na=False)]
    if len(substr_matches) >= limit:
        return substr_matches.head(limit)

    # Fall back to fuzzy matching for typos, merged with substring hits
    all_addresses = lookup["ADDRESS"].astype(str).tolist()
    close = get_close_matches(query, all_addresses, n=limit, cutoff=0.6)
    fuzzy_matches = lookup[lookup["ADDRESS"].isin(close)]
    combined = pd.concat([substr_matches, fuzzy_matches]).drop_duplicates(subset=["FEEDER_NAME", "ADDRESS"])
    return combined.head(limit)


def search_feeders(lookup: pd.DataFrame, query: str = "") -> list:
    feeders = sorted(lookup["FEEDER_NAME"].unique().tolist())
    if not query:
        return feeders
    q = query.strip().lower()
    return [f for f in feeders if q in f.lower()]

#tests/test_filter_by_date.py

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from freezegun import freeze_time

from preprocess_openagenda import filter_by_date, parse_french_datetime_series


# ----------------------------------------------------------------------
# Helper to build test rows
# ----------------------------------------------------------------------
def make_df(first_begin, first_end, last_begin, last_end):
    return pd.DataFrame([{
        "firstdate_begin": first_begin,
        "firstdate_end": first_end,
        "lastdate_begin": last_begin,
        "lastdate_end": last_end,
    }])

def to_timestamp(dt):
    """Convert FakeDatetime → pandas Timestamp (UTC)."""
    if isinstance(dt, datetime):
        return pd.Timestamp(dt).tz_convert("UTC")
    return dt

# ----------------------------------------------------------------------
# Frozen reference time for all recency checks
# Let's freeze at 2025-01-01 12:00 UTC
# ----------------------------------------------------------------------
FROZEN_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_YEAR_AGO = FROZEN_NOW - timedelta(days=365)


# ----------------------------------------------------------------------
# Full scenario matrix
# ----------------------------------------------------------------------
SCENARIOS = [

    # A — All four dates present
    ("all_recent_pass",
     "1 décembre 2024", "2 décembre 2024",
     "3 décembre 2024", "4 décembre 2024",
     True),

    ("start_recent_end_old_fail",
     "1 décembre 2024", "2 décembre 2024",
     "1 janvier 2020", "1 janvier 2020",
     False),

    ("start_old_end_recent_pass",
     "1 janvier 2020", "1 janvier 2020",
     "1 décembre 2024", "2 décembre 2024",
     True),

    # B — Only firstdate_* exist
    ("first_only_recent_pass",
     "1 décembre 2024", "2 décembre 2024",
     None, None,
     True),

    ("first_only_end_old_fail",
     "1 décembre 2024", "1 janvier 2020",
     None, None,
     False),

    ("first_begin_recent_no_end_pass",
     "1 décembre 2024", None,
     None, None,
     True),

    ("first_begin_old_no_end_fail",
     "1 janvier 2020", None,
     None, None,
     False),

    # C — Only lastdate_* exist
    ("last_end_recent_pass",
     None, None,
     "1 décembre 2024", "2 décembre 2024",
     True),

    ("last_end_old_fail",
     None, None,
     "1 janvier 2020", "1 janvier 2020",
     False),

    ("last_begin_recent_no_end_pass",
     None, None,
     "1 décembre 2024", None,
     True),

    ("last_begin_old_no_end_fail",
     None, None,
     "1 janvier 2020", None,
     False),

    # D — Mixed priority rules
    ("priority_first_begin_wins",
     "1 décembre 2024", None,
     "1 janvier 2020", None,
     True),

    ("priority_last_begin_used_if_first_missing",
     None, None,
     "1 décembre 2024", None,
     True),

    ("priority_last_end_overrides_first_end_old_fails",
     "1 décembre 2024", "2 décembre 2024",
     "1 janvier 2020", "1 janvier 2020",
     False),

    ("fallback_to_first_end_when_last_missing",
     "1 décembre 2024", "2 décembre 2024",
     "1 janvier 2020", None,
     True),

    # E — All missing
    ("all_missing_fail",
     None, None, None, None,
     False),

    ("mixed_invalids_first_begin_only_recent",
     "1 décembre 2024", None,
     None, "INVALID",
     True),

    # F — Boundary cases
    ("end_equals_one_year_ago_pass",
     None, ONE_YEAR_AGO.strftime("%d %B %Y"),
     None, None,
     True),

    ("start_equals_one_year_ago_no_end_pass",
     ONE_YEAR_AGO.strftime("%d %B %Y"), None,
     None, None,
     True),

    ("end_just_before_cutoff_fail",
     None, (ONE_YEAR_AGO - timedelta(days=1)).strftime("%d %B %Y"),
     None, None,
     False),

    ("start_just_before_cutoff_no_end_fail",
     (ONE_YEAR_AGO - timedelta(days=1)).strftime("%d %B %Y"), None,
     None, None,
     False),

    # G — DST-sensitive dates
    ("dst_spring_forward_pass",
     "30 mars 2025 02:30", None,
     None, None,
     True),

    ("dst_fall_backward_pass",
     "26 octobre 2025 02:30", None,
     None, None,
     True),

    # H — Future dates
    ("future_start_no_end_pass",
     "1 janvier 2030", None,
     None, None,
     True),

    ("future_end_pass_even_with_old_start",
     "1 janvier 2020", None,
     None, "1 janvier 2030",
     True),

    ("future_end_old_start_pass",
     "1 janvier 2020", "1 janvier 2030",
     None, None,
     True),
]


# ----------------------------------------------------------------------
# Main parametrized test
# ----------------------------------------------------------------------
@freeze_time(FROZEN_NOW)
@pytest.mark.parametrize(
    "name,first_begin,first_end,last_begin,last_end,expected",
    SCENARIOS
)
def test_filter_by_date(name, first_begin, first_end, last_begin, last_end, expected):
    frozen_ts = to_timestamp(datetime.now(timezone.utc))  # convert FakeDatetime → Timestamp
    df = make_df(first_begin, first_end, last_begin, last_end)

    filtered = filter_by_date(df, now=frozen_ts)          # inject real timestamp

    assert (len(filtered) == 1) == expected
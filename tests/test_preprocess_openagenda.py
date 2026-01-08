import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

# Ensure this import matches your file structure
from preprocess_openagenda import (
    parse_french_datetime_series,
    safe_json_parse,
    deserialize_json_fields,
    drop_unnecessary_columns,
    build_rag_document,
    generate_rag_documents,
)

# -------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------

@pytest.fixture
def now():
    """Returns a fixed 'now' to ensure tests are deterministic."""
    return datetime.now(timezone.utc)

@pytest.fixture
def sample_events(now):
    """
    DataFrame simulating comprehensive real-world scenarios for filter_by_date.
    Includes recurring events and boundary testing.
    """
    one_year_ago = now - timedelta(days=365)
    
    return pd.DataFrame([
        # ---------------------------------------------------------
        # BASIC CASES
        # ---------------------------------------------------------
        # [Index 0] ✅ RECENT START: Started 5 days ago. No end date. 
        # Logic: Should be KEPT (Start > 1 year ago).
        {
            "uid": "0",
            "firstdate_begin": (now - timedelta(days=5)).isoformat(),
            "firstdate_end": "", "lastdate_begin": "", "lastdate_end": ""
        },
        # [Index 1] ❌ TOO OLD: Started 13 months ago. No end date.
        # Logic: Should be DROPPED.
        {
            "uid": "1",
            "firstdate_begin": (one_year_ago - timedelta(days=30)).isoformat(),
            "firstdate_end": "", "lastdate_begin": "", "lastdate_end": ""
        },
        
        # ---------------------------------------------------------
        # MISSING DATA LOGIC
        # ---------------------------------------------------------
        # [Index 2] ✅ START RECENT, END MISSING: Implicitly active.
        # Logic: Should be KEPT.
        {
            "uid": "2",
            "firstdate_begin": (now - timedelta(days=20)).isoformat(),
            "firstdate_end": None, "lastdate_begin": None, "lastdate_end": None
        },
        # [Index 3] ✅ START MISSING, END RECENT: Ends 2 days ago.
        # Logic: Should be KEPT (End > 1 year ago).
        {
            "uid": "3",
            "firstdate_begin": None, "firstdate_end": None, "lastdate_begin": None,
            "lastdate_end": (now - timedelta(days=2)).isoformat()
        },
        # [Index 4] ❌ NO DATES: All empty strings.
        # Logic: Should be DROPPED.
        {
            "uid": "4",
            "firstdate_begin": "", "firstdate_end": "", "lastdate_begin": "", "lastdate_end": ""
        },
        # [Index 5] ❌ INVALID DATES: Garbage data + Ancient End date.
        # Logic: Should be DROPPED.
        {
            "uid": "5",
            "firstdate_begin": "not-a-date", "firstdate_end": "not-a-date",
            "lastdate_begin": "not-a-date", 
            "lastdate_end": (one_year_ago - timedelta(days=50)).isoformat()
        },

        # ---------------------------------------------------------
        # ADVANCED PRIORITY & BOUNDARY CASES (ADDED)
        # ---------------------------------------------------------
        # [Index 6] ✅ RECURRING EVENT: 
        # firstdate (start) was 2 years ago (OLD), but lastdate (end) is in Future.
        # Logic: Should be KEPT. This tests if `lastdate_end` overrides `firstdate`.
        {
            "uid": "6",
            "firstdate_begin": (now - timedelta(days=700)).isoformat(),
            "firstdate_end": (now - timedelta(days=699)).isoformat(),
            "lastdate_begin": (now + timedelta(days=10)).isoformat(),
            "lastdate_end": (now + timedelta(days=10)).isoformat() # FUTURE
        },
        # [Index 7] ✅ EXACT BOUNDARY: Ends exactly 365 days ago (to the second).
        # Logic: Should be KEPT (>= operator).
        {
            "uid": "7",
            "firstdate_begin": "", "firstdate_end": "", "lastdate_begin": "",
            "lastdate_end": one_year_ago.isoformat()
        },
    ])

# -------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------

def test_parse_french_datetime_series():
    series = pd.Series(["1 janvier 2024 10:00", "not a date", None, "   "])
    out = parse_french_datetime_series(series)

    assert isinstance(out[0], pd.Timestamp)
    assert out[0].month == 1
    assert pd.isna(out[1]) # Garbage
    assert pd.isna(out[2]) # None
    assert pd.isna(out[3]) # Whitespace

def test_safe_json_parse():
    # Valid JSON
    assert safe_json_parse('{"a": 1}') == {"a": 1}
    # Already Dict
    assert safe_json_parse({"x": 2}) == {"x": 2}
    # None/Empty
    assert safe_json_parse(None) is None
    # Invalid JSON (should return None, not crash)
    assert safe_json_parse('{broken_json:') is None

def test_deserialize_json_fields():
    df = pd.DataFrame({
        "status": ['{"label": {"fr": "Programmé"}}'],
        "attendancemode": ['{"label": {"fr": "Sur place"}}'],
        "timings": ['[{"begin": "2025-03-01"}]'],
        "registration": ['[{"type": "link", "value": "https://test.com"}, {"type": "phone", "value": "0102030405"}]']
    })

    out = deserialize_json_fields(df)

    # Check extraction logic
    assert out.loc[0, "status_label_fr"] == "Programmé"
    assert out.loc[0, "attendancemode_fr"] == "Sur place"
    
    # Check registration formatting (capitalization + newlines)
    reg_contact = out.loc[0, "registration_contact"]
    assert "Link: https://test.com" in reg_contact
    assert "Phone: 0102030405" in reg_contact
    assert "\n" in reg_contact

def test_drop_unnecessary_columns():
    df = pd.DataFrame({
        "title_fr": ["A"],
        "location_coordinates": ["X"], # Should drop
        "status": ["Y"],               # Should drop
        "category": ["Z"],             # Should drop
        "unknown_col": ["KeepMe"]      # Should keep
    })

    out = drop_unnecessary_columns(df)

    assert "location_coordinates" not in out.columns
    assert "status" not in out.columns
    assert "title_fr" in out.columns
    assert "unknown_col" in out.columns

def test_build_rag_document():
    # Simulate a realistic OpenAgenda event row
    row = pd.Series({
        "uid": "123",
        "title_fr": "Concert Test",
        "description_fr": "Une description",
        "longdescription_fr": "Longue description",
        "conditions_fr": "Gratuit",

        "age_min": 10,
        "age_max": 18,

        "keywords_fr": ["musique", "live"],
        "daterange_fr": "Vendredi soir",
        "timings": "18h - 20h",
        "attendancemode_fr": "Sur place",
        "onlineaccesslink": "",
        "registration_contact": "Link: https://test.com",

        # REAL OpenAgenda-like location data
        "location_name": "Salle X",
        "location_address": "10 rue Y",
        "location_postalcode": "75000",
        "location_city": "Paris",
        "location_department": "75",
        "location_region": "IDF",

        # REALISTIC coordinates (OpenAgenda JSON format)
        "location_coordinates": {"lat": 48.85, "lon": 2.35},

        "location_access_fr": "Métro",
        "accessibility_label_fr": "Accessible",
        "originagenda_title": "Agenda test",
        "status_label_fr": "Programmé",

        # Image priority: should select image first
        "image": "http://img.com/cover.jpg",
        "originalimage": "",
        "location_image": ""
    })

    rag = build_rag_document(row)

    # --- Assertions ---

    # Core content
    assert "Titre : Concert Test" in rag
    assert "Description : Une description" in rag
    assert "Description longue : Longue description" in rag
    assert "Détail des conditions : Gratuit" in rag

    # Age
    assert "Public : De 10 à 18 ans" in rag

    # Keywords
    assert "Mots clés : musique, live" in rag

    # Google Maps link (MUST be generated)
    expected_maps = "Plan d'accès : https://www.google.com/maps/search/?api=1&query=48.85,2.35"
    assert expected_maps in rag

    # Correct image priority
    assert "Image Cover : http://img.com/cover.jpg" in rag

    # Status
    assert "Statut : Programmé" in rag

    # Location formatting
    assert "Localisation : Salle X — 10 rue Y — 75000 Paris — 75 (IDF)" in rag


def test_generate_rag_documents():
    # Minimal DF for generator test
    df = pd.DataFrame([{
        "uid": "001",
        "title_fr": "Event A",
        "description_fr": "Desc",
        "longdescription_fr": "",
        "conditions_fr": "",
        "keywords_fr": [],
        "daterange_fr": "",
        "timings": "",
        "location_name": "Loc",
        "location_address": "",
        "location_postalcode": "",
        "location_city": "",
        "location_department": "",
        "location_region": "",
        "image": "",
        "originalimage": "",
        "location_image": "",
        "onlineaccesslink": "",
        "registration_contact": "",
        "location_coordinates": "",
        "location_access_fr": "",
        "accessibility_label_fr": "",
        "originagenda_title": "",
        "attendancemode_fr": ""
    }])

    # Convert generator to list to inspect
    results = list(generate_rag_documents(df))

    assert len(results) == 1
    assert results[0]["uid"] == "001"
    assert "Titre : Event A" in results[0]["text"]
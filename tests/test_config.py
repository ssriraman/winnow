"""Keep/reject decision logic in config dataclasses."""

import pandas as pd

from winnow.config import LogCullCriteria, TechnicalCriteria


def test_technical_criteria_is_and_of_all_conditions():
    c = TechnicalCriteria()  # focus>500, shake>=20, over<=0.05

    assert c.keep(focus=600, shake=25, over=0.01, under=0.0) is True
    assert c.keep(focus=400, shake=25, over=0.01, under=0.0) is False  # soft focus
    assert c.keep(focus=600, shake=10, over=0.01, under=0.0) is False  # shaky
    assert c.keep(focus=600, shake=25, over=0.20, under=0.0) is False  # blown highlights


def test_log_cull_criteria_mask_is_or_of_two_clauses():
    c = LogCullCriteria()  # (focus>350 & shake>19) | (over<0.05 & under<0.30)
    df = pd.DataFrame(
        {
            "focus": [400, 300, 300],
            "shake": [25, 25, 5],
            "over": [0.9, 0.01, 0.9],
            "under": [0.9, 0.10, 0.9],
        }
    )
    # row0: sharp clause; row1: exposure clause; row2: neither
    assert list(c.mask(df)) == [True, True, False]

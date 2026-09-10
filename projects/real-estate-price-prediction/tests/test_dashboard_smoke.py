"""Headless smoke tests for dashboard/app.py via streamlit's AppTest.

Not a UI test — just guards that the prediction page renders and runs a
prediction end-to-end (including the geo-feature and low-support-segment
paths) without raising, against whatever model models/current_model.json
currently points at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP = str(_PROJECT_ROOT / "dashboard" / "app.py")


def _fresh() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=90).run()
    assert not at.exception, f"dashboard failed to start: {at.exception}"
    return at


def _click_calculate(at: AppTest) -> AppTest:
    btn = [b for b in at.button if "Рассчитать" in b.label]
    assert btn, "no 'Рассчитать' button found"
    btn[0].click().run()
    assert not at.exception, f"prediction raised: {at.exception}"
    return at


def _set_city(at: AppTest, city: str) -> None:
    for s in at.selectbox:
        if s.label == "Город":
            s.set_value(city)


def test_dashboard_starts_and_predicts_without_coords():
    at = _fresh()
    _set_city(at, "Москва")
    at.run()
    _click_calculate(at)
    # a price metric must be shown
    assert any("стоимость" in (m.label or "").lower() for m in at.metric)


def test_dashboard_low_support_segment_shows_warning():
    at = _fresh()
    _set_city(at, "Москва")
    for s in at.selectbox:
        if s.label == "Тип объекта":
            s.set_value("Комната")
    at.run()
    _click_calculate(at)
    assert any(
        "ограниченные данные" in w.value.lower() for w in at.warning
    ), "expected a limited-data warning for 'Комната'"


def test_dashboard_geo_model_uses_coordinates():
    """Only meaningful when the current model is the geo model; skips otherwise."""
    at = _fresh()
    # geo model -> the coordinates checkbox exists
    coord_cb = [c for c in at.checkbox if c.label.lower().startswith("указать координаты")]
    if not coord_cb:
        pytest.skip("current model has no geo features — no coordinate input on the page")
    _set_city(at, "Москва")
    at.run()
    for c in at.checkbox:
        if c.label.lower().startswith("указать координаты"):
            c.set_value(True)
    at.run()
    lat = [n for n in at.number_input if n.label == "Широта"]
    lon = [n for n in at.number_input if n.label == "Долгота"]
    assert lat and lon
    lat[0].set_value(55.7558)
    lon[0].set_value(37.6173)
    at.run()
    _click_calculate(at)
    assert any(
        "геопризнаки использованы" in s.value.lower() for s in at.success
    ), "expected the 'geo features used' confirmation"

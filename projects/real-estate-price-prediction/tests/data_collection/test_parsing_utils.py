"""Unit tests for src.data_collection.parsing_utils — pure functions, no
network, no BeautifulSoup/Selenium objects involved."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.parsing_utils import (  # noqa: E402
    parse_area,
    parse_floor_info,
    parse_price,
    parse_rooms,
    random_user_agent,
)


class TestParsePrice:
    def test_price_with_spaces_and_rouble_sign(self):
        assert parse_price("13 200 000 ₽") == 13_200_000.0

    def test_price_with_nbsp_and_word_rub(self):
        assert parse_price("5\xa0100\xa0000 руб.") == 5_100_000.0

    def test_price_empty_string(self):
        assert parse_price("") is None

    def test_price_none_like_input(self):
        assert parse_price(None) is None  # type: ignore[arg-type]

    def test_price_no_digits(self):
        assert parse_price("цена по запросу") is None


class TestParseRooms:
    def test_studio_russian(self):
        assert parse_rooms("Студия, 21 м²") == 0

    def test_studio_english(self):
        assert parse_rooms("Studio apartment") == 0

    def test_two_room_dot_komn(self):
        assert parse_rooms("2-комн. квартира, 54.3 м²") == 2

    def test_dash_k_abbreviation(self):
        assert parse_rooms("3-к квартира") == 3

    def test_no_room_info(self):
        assert parse_rooms("Апартаменты у моря") is None

    def test_empty_string(self):
        assert parse_rooms("") is None


class TestParseArea:
    def test_comma_decimal(self):
        assert parse_area("45,5 м²") == 45.5

    def test_dot_decimal_kvm(self):
        assert parse_area("45.5 кв.м") == 45.5

    def test_no_area(self):
        assert parse_area("нет данных") is None


class TestParseFloorInfo:
    def test_floor_over_total(self):
        assert parse_floor_info("7/16 эт.") == (7, 16)

    def test_floor_only(self):
        assert parse_floor_info("3 эт.") == (3, None)

    def test_empty_string(self):
        assert parse_floor_info("") == (None, None)

    def test_no_floor_pattern(self):
        assert parse_floor_info("рядом с метро") == (None, None)


def test_random_user_agent_returns_nonempty_string():
    ua = random_user_agent()
    assert isinstance(ua, str)
    assert len(ua) > 10

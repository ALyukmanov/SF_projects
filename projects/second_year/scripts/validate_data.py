"""Проверка сырых данных. Запуск: python scripts/validate_data.py"""
import sys

import _bootstrap  # noqa: F401

from second_year.data.validate import main

if __name__ == "__main__":
    sys.exit(main())

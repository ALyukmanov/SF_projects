"""
Telegram bot for real estate price prediction.
Uses python-telegram-bot v20+ with async handlers.

Setup:
    1. pip install python-telegram-bot python-dotenv
    2. Set TELEGRAM_BOT_TOKEN in .env or as an environment variable.
    3. Run: python telegram_bot/bot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
_BOT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BOT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load .env before anything else
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _env_path = _PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv optional; token can be set directly in the environment

# ---------------------------------------------------------------------------
# Graceful import of python-telegram-bot
# ---------------------------------------------------------------------------
try:
    from telegram import (
        ReplyKeyboardMarkup,
        ReplyKeyboardRemove,
        Update,
    )
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        ConversationHandler,
        MessageHandler,
        filters,
    )
except ImportError:
    print(
        "\n[ERROR] python-telegram-bot is not installed.\n"
        "Run the following command and try again:\n\n"
        "    pip install python-telegram-bot python-dotenv\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------------
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger("telegram_bot")

# ---------------------------------------------------------------------------
# Constants – conversation states
# ---------------------------------------------------------------------------
(
    STATE_CITY,
    STATE_ROOMS,
    STATE_AREA,
    STATE_FLOOR,
    STATE_FLOORS_TOTAL,
) = range(5)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
_CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Екатеринбург",
    "Новосибирск",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
]

_ROOMS_OPTIONS = ["Студия", "1", "2", "3", "4+"]

# Map display names → predictor city strings
_CITY_MAP: dict[str, str] = {
    "Москва": "москва",
    "Санкт-Петербург": "санкт-петербург",
    "Екатеринбург": "екатеринбург",
    "Новосибирск": "новосибирск",
    "Казань": "казань",
    "Нижний Новгород": "нижний новгород",
    "Самара": "самара",
    "Краснодар": "краснодар",
}

_ROOMS_MAP: dict[str, int] = {
    "студия": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4+": 4,
}

# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------


def _city_keyboard() -> ReplyKeyboardMarkup:
    """4-column grid of city names."""
    row1 = _CITIES[:4]
    row2 = _CITIES[4:]
    return ReplyKeyboardMarkup(
        [row1, row2],
        one_time_keyboard=True,
        resize_keyboard=True,
    )


def _rooms_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [_ROOMS_OPTIONS],
        one_time_keyboard=True,
        resize_keyboard=True,
    )


def _remove_keyboard() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove()


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _format_prediction(data: dict, mode_note: str) -> str:
    price = data["price"]
    price_min = data["price_min"]
    price_max = data["price_max"]
    city = data["city_display"]
    rooms = data["rooms_display"]
    area = data["area"]
    floor = data["floor"]
    floors_total = data["floors_total"]
    ppm = round(price / area) if area else 0

    return (
        "🏠 *Прогноз цены*\n\n"
        f"📍 Город: {city}\n"
        f"🛏 Комнат: {rooms}\n"
        f"📐 Площадь: {area} кв.м\n"
        f"🏢 Этаж: {floor}/{floors_total}\n\n"
        f"💰 *Цена: {price:,} ₽*\n"
        f"📊 Диапазон: {price_min:,} — {price_max:,} ₽\n"
        f"💵 За кв.м: {ppm:,} ₽\n\n"
        f"_{mode_note}_"
    ).replace(",", " ")  # Russian thousands separator


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    text = (
        "👋 Привет! Я бот для прогнозирования цен на недвижимость в России.\n\n"
        "Я умею оценивать стоимость квартиры на основе её характеристик, "
        "используя модели машинного обучения.\n\n"
        "Команды:\n"
        "• /predict — получить прогноз цены\n"
        "• /help — справка\n"
        "• /cancel — отменить текущую операцию"
    )
    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help message."""
    text = (
        "ℹ️ *Справка*\n\n"
        "Используйте /predict, чтобы запустить диалог оценки квартиры.\n\n"
        "Бот последовательно спросит:\n"
        "1. Город\n"
        "2. Количество комнат\n"
        "3. Общую площадь (кв.м)\n"
        "4. Этаж квартиры\n"
        "5. Этажность дома\n\n"
        "После ввода всех данных будет показан прогноз цены "
        "с доверительным диапазоном и ценой за кв.м.\n\n"
        "Для отмены введите /cancel."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Операция отменена. Введите /predict, чтобы начать заново.",
        reply_markup=_remove_keyboard(),
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Conversation steps
# ---------------------------------------------------------------------------


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point: ask for city."""
    context.user_data.clear()
    await update.message.reply_text(
        "🏙 Выберите город:",
        reply_markup=_city_keyboard(),
    )
    return STATE_CITY


async def receive_city(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store city, ask for rooms."""
    city_display = update.message.text.strip()
    if city_display not in _CITY_MAP:
        await update.message.reply_text(
            "Пожалуйста, выберите город из предложенного списка 👆",
            reply_markup=_city_keyboard(),
        )
        return STATE_CITY

    context.user_data["city_display"] = city_display
    context.user_data["city"] = _CITY_MAP[city_display]

    await update.message.reply_text(
        "🛏 Количество комнат:",
        reply_markup=_rooms_keyboard(),
    )
    return STATE_ROOMS


async def receive_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store rooms, ask for area."""
    raw = update.message.text.strip().lower()
    rooms = _ROOMS_MAP.get(raw)
    if rooms is None:
        await update.message.reply_text(
            "Пожалуйста, выберите количество комнат из списка 👆",
            reply_markup=_rooms_keyboard(),
        )
        return STATE_ROOMS

    context.user_data["rooms"] = rooms
    context.user_data["rooms_display"] = update.message.text.strip()

    await update.message.reply_text(
        "📐 Введите общую площадь квартиры в кв.м (например: 55.5):",
        reply_markup=_remove_keyboard(),
    )
    return STATE_AREA


async def receive_area(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store total_area, ask for floor."""
    raw = update.message.text.strip().replace(",", ".")
    try:
        area = float(raw)
        if not (5.0 <= area <= 1000.0):
            raise ValueError("out of range")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Введите корректную площадь (число от 5 до 1000), например: 55.5"
        )
        return STATE_AREA

    context.user_data["area"] = area

    await update.message.reply_text("🏢 Введите номер этажа квартиры (например: 5):")
    return STATE_FLOOR


async def receive_floor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store floor, ask for floors_total."""
    raw = update.message.text.strip()
    try:
        floor = int(raw)
        if not (1 <= floor <= 200):
            raise ValueError("out of range")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Введите корректный номер этажа (целое число от 1 до 200), например: 5"
        )
        return STATE_FLOOR

    context.user_data["floor"] = floor

    await update.message.reply_text("🏗 Введите этажность дома (всего этажей, например: 17):")
    return STATE_FLOORS_TOTAL


async def receive_floors_total(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store floors_total, run prediction, send result."""
    raw = update.message.text.strip()
    try:
        floors_total = int(raw)
        if not (1 <= floors_total <= 200):
            raise ValueError("out of range")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Введите корректную этажность (целое число от 1 до 200), например: 17"
        )
        return STATE_FLOORS_TOTAL

    floor = context.user_data.get("floor", 1)
    if floors_total < floor:
        await update.message.reply_text(
            f"⚠️ Этажность дома ({floors_total}) не может быть меньше этажа квартиры ({floor}). "
            "Введите корректную этажность:"
        )
        return STATE_FLOORS_TOTAL

    context.user_data["floors_total"] = floors_total

    # ------------------------------------------------------------------
    # Run prediction
    # ------------------------------------------------------------------
    predictor: Predictor = context.bot_data["predictor"]

    features = {
        "city": context.user_data["city"],
        "rooms": context.user_data["rooms"],
        "total_area": context.user_data["area"],
        "floor": context.user_data["floor"],
        "floors_total": floors_total,
    }

    try:
        result = predictor.predict(features)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        await update.message.reply_text(
            "😔 Произошла ошибка при расчёте прогноза. Попробуйте снова: /predict"
        )
        context.user_data.clear()
        return ConversationHandler.END

    mode = result.get("mode", "demo")
    if mode == "model":
        confidence = result.get("confidence", 0.85)
        mode_note = f"Прогноз модели ML (достоверность {confidence:.0%})"
    else:
        mode_note = "Демо-режим: эвристическая оценка без обученной модели"

    display_data = {
        "price": result["price"],
        "price_min": result["price_min"],
        "price_max": result["price_max"],
        "city_display": context.user_data["city_display"],
        "rooms_display": context.user_data["rooms_display"],
        "area": context.user_data["area"],
        "floor": context.user_data["floor"],
        "floors_total": floors_total,
    }

    message = _format_prediction(display_data, mode_note)

    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_remove_keyboard(),
    )

    context.user_data.clear()
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def build_application(token: str) -> Application:
    """Build and configure the Application."""
    app = Application.builder().token(token).build()

    # Shared predictor (loaded once; falls back to DEMO automatically)
    predictor = Predictor(model_path=str(_PROJECT_ROOT / "models"))
    predictor.load()
    app.bot_data["predictor"] = predictor

    if predictor.is_ready():
        logger.info("Predictor loaded: model mode active.")
    else:
        logger.info("Predictor in DEMO mode (no trained model found).")

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            STATE_CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_city)],
            STATE_ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_rooms)],
            STATE_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_area)],
            STATE_FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_floor)],
            STATE_FLOORS_TOTAL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_floors_total)
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(conv_handler)
    app.add_error_handler(error_handler)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(
            "\n[ERROR] TELEGRAM_BOT_TOKEN is not set.\n\n"
            "Options:\n"
            "  1. Create a .env file in the project root and add:\n"
            "         TELEGRAM_BOT_TOKEN=your_token_here\n"
            "  2. Or export it as an environment variable:\n"
            "         export TELEGRAM_BOT_TOKEN=your_token_here\n\n"
            "You can obtain a token from @BotFather on Telegram.\n"
        )
        sys.exit(1)

    logger.info("Starting Telegram bot...")
    app = build_application(token)

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()

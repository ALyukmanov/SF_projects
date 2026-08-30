.PHONY: up down logs train test traffic smoke build ps clean-volumes

up:
	docker compose up --build -d

build:
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

train:
	python -m app.training

test:
	python -m pytest -q

traffic:
	python scripts/generate_requests.py --count 200 --delay 0.1

smoke:
	python scripts/smoke_test.py

clean-volumes:
	docker compose down -v

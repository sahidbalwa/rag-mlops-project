.PHONY: install run test docker-up docker-down clean

install:
	pip install -r requirements.txt

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

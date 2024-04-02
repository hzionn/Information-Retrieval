test:
	pytest tests

cov:
	pytest --cov=ir tests

run:
	python main.py

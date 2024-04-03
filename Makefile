run:
	python main.py

vec:
	python -m ir.vectorspace

test:
	pytest tests

cov:
	pytest --cov=ir tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage* .DS_Store *_UML.txt
	rm -rf */__pycache__ */.DS_Store */*_UML.txt

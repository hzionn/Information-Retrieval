run:
	python main.py --sample-size 500

vec:
	python -m ir.vectorspace

test:
	pytest tests

cov:
	pytest --cov=ir tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage* .DS_Store *_UML.txt
	rm -rf */__pycache__ */.DS_Store */*_UML.txt

install:
	pip install -U -r requirements.txt

test:
	py.test

yapf:
	find . -type f -name "*.py" | xargs yapf -i

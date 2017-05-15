install_deps:
	pip install -U -r requirements.txt

install:
	pip install ./

test:
	python -m pytest tests/

yapf:
	find . -type f -name "*.py" | xargs yapf -i

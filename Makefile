all: build install

build:
	python setup.py build_ext --inplace

install:
	python setup.py install

clean:
	python setup.py clean --all
	rm -rf build dist *.egg-info
	find . -name '*.so' -type f -delete
	find . -name '*.pyc' -type f -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +

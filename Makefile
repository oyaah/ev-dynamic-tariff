.PHONY: install run test notebooks docker clean

install:
	pip install -r requirements.txt

run:            ## validate data, run agents, save models + all results
	python -m src.run_pipeline

test:
	python -m pytest tests/ -q

notebooks:      ## regenerate and execute the 5 notebooks
	python build_notebooks.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker:         ## build the reproducible image
	docker build -t ev-tariff .

clean:
	rm -rf __pycache__ src/__pycache__ src/agents/__pycache__ tests/__pycache__ .pytest_cache models

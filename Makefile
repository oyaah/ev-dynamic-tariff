.PHONY: install run test notebooks clean

install:
	pip install -r requirements.txt

run:            ## build unified base + all results into outputs/
	python -m src.run_pipeline

test:
	python -m pytest tests/ -q

notebooks:      ## regenerate and execute the 5 notebooks
	python build_notebooks.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf __pycache__ src/__pycache__ src/agents/__pycache__ tests/__pycache__ .pytest_cache

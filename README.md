**Install requirements (need python 3.11)**
pip install -r requirements.txt

**Run random motions with GUI**
python test_env.py

clone the OGbench repo into your project directory, make sure its in gitignore so you dont commit it to this repo

install the new requirements (some are in OGbench's requirement.txt I think)

Run test_ogbench.py

**Running OGBench + HIQL**

This project depends on the local `ogbench` repository.

Because the OGBench Python package is nested (`ogbench/ogbench`),
you must add the repo root to your `PYTHONPATH` before running:

```bash
export PYTHONPATH="$(pwd)/ogbench:$PYTHONPATH"
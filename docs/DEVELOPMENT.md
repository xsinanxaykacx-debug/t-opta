# Development

Install: python -m pip install -r requirements.txt

Test: python -m pytest

Invalid/missing timestamps are rejected, not silently discarded. test_days controls holdout width; step_days controls the next fold start. Rows at the fixed cutoff belong to test. Run the full suite before every push; CI repeats pytest on push and pull request.

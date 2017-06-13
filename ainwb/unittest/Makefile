# These commands are for executing the unittests on Python 2.7 and Python 3+
test-python2:
	python ./test_nwb.py

test-python3:
	python3 ./test_nwb.py

test-python2-silent:
	@echo "piping output to 'output.txt'"
	@python ./test_nwb.py > ./output.txt

test-python3-silent:
	@echo "piping output to 'output.txt'"
	@python3 ./test_nwb.py > ./output.txt

# These commands are for checking, modifying, and resetting the temp directory
check-temp-dir:
	@python -c "import tempfile; print('The test directory is %s' % tempfile.gettempdir())"

setting-temp-dir:
	@echo "GNU make cannot set environment variables on its own."
	@echo "Run 'export TEMP=$$(pwd)/' to have test files saved in the current directory"
	@echo "Run 'unset TEMP' to return to default location"

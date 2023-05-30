# Script to run CI checks locally
echo -e "RUNNING CHECKS (`date`)" && \
echo -e "\nRunning black..." && black . --check && \
echo -e "\nRunning isort..." && isort . --check && \
echo -e "\nRunning pyright..." && pyright && \
echo -e "\nRunning pylint..." && pylint $(git ls-files 'planetmapper/*.py') setup.py check_release_version.py && \
echo -e "\nRunning tests..." && python -m coverage run -m unittest discover -s tests -v && python -m coverage report && \
echo -e "\nALL DONE"

# Script to run GitHub Actions checks locally.
# This isn't a 1:1 copy of the GitHub workflow (e.g. only runs on one OS & Python 
# version) but it's generally close enough to catch most issues before pushing.

# The dependencies for these checks can be installed by running:
# pip install -r dev_requirements.txt

echo -e "RUNNING CHECKS (`date`)"

# Collect python type stubs so that pyright works with matplotlib
if [ ! -d "python-type-stubs" ]; then
    echo -e "\nCloning python-type-stubs for use with pyright..."
    git clone https://github.com/microsoft/python-type-stubs python-type-stubs
fi

echo -e "\nUpdating python-type-stubs (for use with pyright)..."
cd python-type-stubs && git pull && cd ..

# Check current environment is consistent with the requirements
echo -e "\nChecking environment is consistent with requirements..."
pip install -r requirements.txt -r dev_requirements.txt --dry-run | grep "Would install" || echo "All requirements are satisfied."

# Allow the docstring check to fail (end line with ";"), all others should cause
# the script to stop (end lines with "&&"), as the docstring check only really
# matters when we are releasing a new version, so it's normal for it to fail when
# the next version number is currently unknown.
echo -e "\nChecking documentation version directives..." && python check_docstring_version_directives.py; \
echo -e "\nRunning black..." && black . --check --diff && \
echo -e "\nRunning isort..." && isort . --check --diff && \
echo -e "\nRunning pyright..." && pyright && \
echo -e "\nRunning pylint..." && pylint $(git ls-files 'planetmapper/*.py') setup.py check_release_version.py && \
echo -e "\nRunning tests..." && python -m coverage run -m unittest discover -s tests && python -m coverage report && python -m coverage html && \
echo -e "\nMaking documentation..." && cd docs && make html && cd .. && \
echo -e "\nALL DONE"

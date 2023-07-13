echo [$(date)]: 'START'
echo [$(date)]: 'Creating conda env with python 3.10'
conda create --prefix ./venv python=3.10 -y
echo [$(date)]: 'activate venv'
source activate ./venv
echo [$(date)]: 'installing dev requirements'
pip install -r requirements.txt
echo [$(date)]: 'Setup END'
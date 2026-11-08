#!/usr/bin/env bash
# Створення віртуального оточення і встановлення залежностей (macOS / Linux).
# Запуск:  bash setup_venv.sh   потім   source .venv/bin/activate
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name aco-sop --display-name "Python (aco-sop)" || true
echo "Готово. Активуйте оточення: source .venv/bin/activate"

#!/bin/zsh
cd -- "${0:A:h}" || exit 1
PYTHON_BIN="${URBAN_SCENE_PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import torch, torchvision, PIL, numpy' >/dev/null 2>&1; then
  echo '缺少运行依赖。请先在此文件夹的终端运行：python3 -m pip install -r requirements-demo.txt'
  read '?按回车退出…'
  exit 1
fi
"$PYTHON_BIN" app.py
if [[ $? -ne 0 ]]; then
  read '?启动失败，按回车退出…'
fi

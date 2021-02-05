pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

#jupyter lab --port=1234 --no-browser
#ssh -N -f -L 1234:localhost:1234 ubuntu@192.249.18.74

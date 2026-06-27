import fairseq_signals

print("fairseq_signals import OK:", fairseq_signals.__version__)

import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: None")

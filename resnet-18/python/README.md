https://huggingface.co/microsoft/resnet-18

```bash
mkdir -p $HOME/models/resnet-18
hf download microsoft/resnet-18 --local-dir $HOME/models/resnet-18 --exclude='*.pth'
hf download huggingface/cats-image --repo-type dataset --local-dir $HOME/dataset/cats-image
```

```bash
uv venv
source .venv/bin/activate
uv sync
```

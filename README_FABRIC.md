# DDP

## Docker setup

```bash
docker build -t kyi-atari .

bash run_docker.sh
```

## Run 1 GPU

```bash
fabric run model --accelerator=cuda --devices=1 --strategy=ddp main.py
```

## Run 2 GPUs

```bash
fabric run model --accelerator=cuda --devices=2 --strategy=ddp main.py
```

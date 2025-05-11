# YOLO Auto Annotation

Using Moondream visual model the script will detect object on specified images and create YOLO dataset for future training.

Currently it supports only **single class**, aka all labels will represent single class. Because that was my requirement for the tool.

https://github.com/user-attachments/assets/92007c2a-5cc4-4f7d-838a-138e4e84fd19

## Run

Currently script uses CUDA, if you don't have NVIDIA GPU then you will need to do some adjustments.

Clone repository:

```bash
git clone https://github.com/roman-koshchei/yolo-auto-annotation.git
```

Sync dependencies using `uv` package manager:

```bash
uv sync
```

Run script with arguments:

```bash
uv run main.py --source {directory with images} --destination {directory where dataset will be saved} --classes {list of classes}
```

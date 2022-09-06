```bash
sh download_dataset.py
```

# Dataset structure
```bash
dataset---train---images---0.jpg
       |        |      |___1.jpg
       |        |
       |        |__masks---0.png
       |        |      |___1.png
       |        |
       |___test___images__...
       |        |
       |        |__masks__...
       |
       |___val___images___...
                |
                |_masks___...

```

```bash
python train.py --root dataset/Figaro_1k_png --batch 16 --num-workers 8
```
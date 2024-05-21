# Web Text Super-Solution(R) based on ttf

## Info

## Train

Make sure there are two folders, `output` and `data`, in the directory.

### Environment

Just run `pip -r requirements.txt` to intall library from pip.

### Data Generation

Just run `python create_img.py` to generate image with kinds of txt for training.

Or you can set your own dataset.

### Train

`python train.py` to train models

### Eval

`python webSTR.py [your path to image]`

For example:

`python webSTR.py .\example.png` to SR your image in path `.\example.png`.

## Contract us

## Relate Work

[TextBSR](https://github.com/csxmli2016/textbsr)

[EasyOCR](https://github.com/JaidedAI/EasyOCR)

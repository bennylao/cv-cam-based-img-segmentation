# cv-CAM-based-img-segmentation

## Noteboooks and Scripts

Experiments and results are also stored in the `notebooks` directory.

### Training

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.train.train_baseline_deeplab
    ```

- baseline Unet model

    ```shell
    python -m src.train.train_baseline_unet
    ```

### Evaluation

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.eval.eval_baseline_deeplab 
    ```

- baseline Unet model

    ```shell
    python -m src.eval.eval_baseline_unet
    ```

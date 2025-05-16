# cv-CAM-based-img-segmentation

## Models

Models weights can be downloaded from the following link:
[model_weights](https://www.icloud.com/iclouddrive/05f4KsOqKw_NywJZlf6qifJ5g#GradCAM-Weakly-Supervision)

## Noteboooks and Scripts

Experiments and results are also stored in the `notebooks` directory.

### Training

- GradCAM++ deeplabv3 resnet50 with self-training and superpixels slic

    ```shell
    python -m src.train.train_cam_slic_deeplab
    ```

- GradCAM++ deeplabv3 resnet50

    ```shell
    python -m src.train.train_cam_deeplab
    ```

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.train.train_baseline_deeplab
    ```

- baseline Unet model

    ```shell
    python -m src.train.train_baseline_unet
    ```

### Evaluation

- GradCAM++ deeplabv3 resnet50 with self-training and superpixels slic

    ```shell
    python -m src.eval.eval_cam_slic_deeplab
    ```

- GradCAM++ deeplabv3 resnet50

    ```shell
    python -m src.eval.eval_cam_deeplab
    ```

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.eval.eval_baseline_deeplab 
    ```

- baseline Unet model

    ```shell
    python -m src.eval.eval_baseline_unet
    ```

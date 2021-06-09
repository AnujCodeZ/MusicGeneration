# Music Generation

Music Generation of piano music using PixelCNN.

## Steps:
1. **Installation**:

    `$ pip install -r requirements.txt`

2. **Getting data**:
    - Get any midi piano files dataset. such as [ADL piano midi](https://github.com/lucasnfe/adl-piano-midi) which I used.

3. **Convert midi files to images**:

    ```
    $ cd data/
    $ python preprocess_img.py <midi files path> <img files path> <repetitions>
    ```
    - img files path: where to save converted images
    - repetitions: how many images from one file.

4. **Train PixelCNN**:

    ```
    $ cd ..
    $ python train.py --data_dir <img files path>
    ```

5. **Generate images**:

    `$ python generate.py`
    - generated image should save as sample.png

6. **Convert back to midi**:

    ```
    $ cd data/
    $ python post_process.py <generated img path>
    ```

> Thanks
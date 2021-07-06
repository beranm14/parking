# Using Auto ML to classify how many parking spaces are there

This is an implementation of the AutoML
[tutorial](https://cloud.google.com/vision/automl/docs/tutorial#python) for getting the count
of parking spaces in front of the house I live in.

## Collecting data

I used old Raspberry Pi with USB cam. Using Raspberry cam is also an option, but I had issues with the length of
the cable. The script is executed via cron:

```bash
*/10  *	* * *	root	bash /home/pi/snap.sh
```

Where `snap.sh` contains following:

```bash
#!/bin/bash
mount | grep "/mnt/parking_photos" >/dev/null || mount /mnt/parking_photos
fswebcam -D 2 -S 20 -F 10 -r 1920x1080 --no-banner `date +"/mnt/parking_fotos/%y%m%d_%H%M%S.jpg"`
```

Folder `parking_photos` is a `NFS` shared volume. You do not need that once you are willing to save the photos
on a local filesystem.

Info about parameters and arguments could be found at
[raspberrypi.org](https://www.raspberrypi.org/documentation/usage/webcams/). In case your use case is the same
I would suggest using at least `-F 10`. I was not interested in the things which moved in the picture and this
helped me to partially blur them away.

## Preprocessing

### Removing dark pictures

Copy pictures captured by Raspberry to `parking_photos_raw` folder. Then execute `filter_dark_raw_photos.sh`
to remove pictures which are just too dark. It will start a lot of shell scripts in parallel. Overall it will
compress the picture to 1x1 pixel and check if brightness is higher than 20:

```
if [ `convert $1 -colorspace hsb -resize 1x1 txt:- | grep -o 'hsb\(.*,.*,.*\)' | grep -o '[0-9]*%)' | grep -o '[0-9]*'` -gt 20 ]; then
  echo $1
else
  rm "$1"
fi
```

### Preparing testing data

This is the most tedious part. Just go trough the pictures and divide them into folders in `data` directory.
Your `data` folder may differ, but I had 1, 2, 3, 4 and i, full. Each of them tells how many parking spaces are
empty. `i` stands for disabled parking space.

### Creating list of pictures

AutoML accepts the list in the format described in the
[documentation](https://cloud.google.com/vision/automl/docs/prepare#csv). Once you upload the pictures in to GCS
bucket, you can use script `data/generate_list.sh` to get the list. It will divide the pictures into 80% of training,
10% for testing and 10% for validation.

## Howto

The steps I took are the same as described in [tutorial](https://cloud.google.com/vision/automl/docs/tutorial#python)

Go to `automl` directory & use:

1) `vision_classification_create_dataset.py` to create the dataset,
2) `import_images_to_dataset.py` to import pictures,
3) `vision_classification_create_model.py` to create model,
4) `list_model_evaluations.py` to see if this all was worth it.

Do not forget to fill `.env` file in `automl` folder.

## Results

Use `python3 vision_classification_predict.py testing_photos/210705_161002.jpg` to evaluate the testing pictures:

`file_path = "testing_photos/210705_101001.jpg"`:

```
Prediction results:
Predicted class name: 2
Predicted class score: 0.44264715909957886
Predicted class name: 3
Predicted class score: 0.4311576187610626
```

`file_path = "testing_photos/210705_140001.jpg"`:

```
Prediction results:
Predicted class name: 2
Predicted class score: 0.5498601794242859
Predicted class name: 3
Predicted class score: 0.4054759740829468
```

`file_path = "testing_photos/210705_161002.jpg"`:

```
Prediction results:
Predicted class name: i
Predicted class score: 0.8651795387268066
```

As you can see, the score is not as high as desirable. Once you need to know if you can park the car,
this number seems to be enough.

## Camera holder

If you need to place the camera somewhere reasonable, you can use 3D printed holder.
It's the [holder](./holder) folder.

## Comparing with Tensorflow implementation

To have something to compare AutoML with, I tried standard Tensoflow
[tutorial](https://www.tensorflow.org/tutorials/images/classification) on image classification. I had to change
few constrains:

 * Large size of the batch did not help with better accuracy
 * Adding image shifting did not help either
 * Enlarging meta arguments like the count of `Conv2D` layers did help
 * Enlarging input image size helped a bit as well

Overall results were sufficient:
```
210705_101001.jpg image most likely belongs to 1 with a 98.84 percent confidence.
210705_140001.jpg image most likely belongs to 3 with a 82.20 percent confidence.
210705_145001.jpg image most likely belongs to i with a 80.34 percent confidence.
210705_153001.jpg image most likely belongs to i with a 100.00 percent confidence.
210705_154002.jpg image most likely belongs to i with a 99.99 percent confidence.
210705_155001.jpg image most likely belongs to i with a 99.72 percent confidence.
210705_160001.jpg image most likely belongs to i with a 99.99 percent confidence.
210705_161002.jpg image most likely belongs to i with a 100.00 percent confidence.
```


# N_similar_images

I used Elbow method to get optimal value of k clusters in the dataset.
I got to know that K=6 is optimal value for the dataset.

segregated the dataset using Resnet based kmeans algofithm into 6 clusters.
`python resnet_kmeans.py --input_dir /dataset/dir --target_dir /segregated/dir`

Train test split
`python train_test_Split.py --data_path=/segregated/dir --test_data_path_to_save=/test/dir --train_ratio=0.7`

I choosed a ResNet50 to train with the segregated data.
`python train_animals.py`

Used trained model to get N similar images as that of input image.
Used already trained weights of ResNet50 model with give 5K dataset
`python classification.py --input_img /data1/prjs/OCR/ocrmypdf/outdir_sub/5/5_52.jpg --N_req_imgs 4 --load_w /home/shahmustafa/Desktop/bottleneck_fc_model_save.h5`
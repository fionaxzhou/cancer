##pipeline

#preprocess
./img_mask_gen.py #geneate image(lung)/mask(nodule) pairs 

#train (use load_data.ImgStream to load image and feed to ImageAugment to do augmentation)
rm config.py
ln -s config_v5.py config.py #v5 is currently a good starting point
./train.py

#prediction

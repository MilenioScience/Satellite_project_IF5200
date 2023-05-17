# Satellite_project_IF5200
Model damage building assessment using HR-Net and Do Counting

This project use python version 3.10.11, numpy version 1.22.4 and torch version 2.0.0+cu118

Dataset available on this website : https://www.xview2.org/
You need to sign up then you can download dataset

This model use HR-Net from this repository : https://github.com/SIAnalytics/dual-hrnet
I added the algorithm code for counting the predicted mask for each damage building classes
Evaluate the model using this evaluation provided by xview2 : https://github.com/DIUx-xView/xView2_scoring

Please install yacs before you do training, testing, or inference by simply write !pip install yacs on your notebook

How to do training : 
!python3 train_net.py --data_dir=./ \
                    --config_path=/content/drive/MyDrive/Satelite_project/dual-hrnet-master/configs/dual-hrnet.yaml \
                    --ckpt_save_dir=/content/drive/MyDrive/Satelite_project/dual-hrnet-master/checkpoints
How to do inference : 
!python3 inference.py

Pretrained weight link : https://drive.google.com/file/d/1wB8I8zw9tQ8adiBBGS5wdHBh2XGf9bck/view

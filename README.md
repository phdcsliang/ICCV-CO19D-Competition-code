# ICCV-CO19D-Competition-code
This is the code base for my paper "A hybrid deep learning framework for Covid-19 detection via 3D Chest CT Images" 
arxiv: TBD
This repo main focus on how to make full use of 3D CT data while avoiding heavy data input.
The 3D CT scans were resampled using a uniform downsampling strategy, the details can be obtained in the dataset/dataloder.py

to use this repo for training

you should first mkdir named 'data' and then put the competition train and valid data into the data folder

which should look like this

-data
--train
----covid
----non-covid
--valid
----covid
----non-covid

then just run train.py


after trained

you can just simply run infer.py
to get the test results.

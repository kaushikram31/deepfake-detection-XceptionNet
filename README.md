# "Deepfake-detection" with Explainable AI
This repo contains the code, results, models for Deep Fake detection with XAI. Course Project for ECE 792 - Advanced Machine Learning

**Project Motive and Description**
* To classify real and fake images, pretrained XceptionNet model is implemented.
* For model interpretability, algorithms such as LIME and Grad-CAM are implemented.
* The objective of this project is to incorporate explainable AI (XAI) techniques to gain insights into the interpretability of the model. The project utilizes state-of-the-art XceptionNet to detect deepfakes, and then employs LIME and GradCam algorithms to visualize and analyze how the model interprets the results.
* Images were used instead of Videos from the dataset and a python script was written to extract images from the videos.
* FaceForensic++ and Celeb-DF Datasets were used for model training and testing

#### Files Description
- ```models```: contains trained models for FaceForensic++ and Celeb-DF datasets
- ```outputs```: contains output results of prediction of real and fake, lime and grad-cam algorithms
- ```plots```: contains plots of accuracy, loss against epochs
- ```CelebDF_XceptionNet_Code.ipynb```: contains code of xception on celebdf dataset
- ```FaceForensics_XceptionNet_Code.ipynb```: contains code of xception on faceforensics dataset
- ```face_extraction_code_from_vidoes.py```: contains code for extraction of faces from the videos
- ```grad_cam.ipynb```: contains the code for gradCAM algorithm
- ```lime_code.ipynb```: contains the code for LIME algorithms
- ```plot_curves.ipynb```: contains the code to plot the graphs
- ```test_predictions.ipynb```: continas the code for predicicion on test dataset

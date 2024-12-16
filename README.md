# DSUA-301-Final Project-Moodify
Most music recommendation systems today rely heavily on user history or listening habits, lacking real-time interaction based on users' current mood or context. 
This project combines real-time emotion detection with music recommendation to create a personalized and context-aware user experience.
We trained emotion detection models on the FER2013 dataset with fine-tuned CNN, VGG16, ResNet and paired them with a music recommendation model based on CNN and RNN-attention architectures.
The system can predict emotions like angry, sad, happy, surprise, and fear, then recommend 10 songs tailored to the detected emotion.

 ## Environment setup
 ### 1. requirements
 - Python >= 3.8
 - Install dependencies with pip:
   
 `pip install -r requirements.txt`

 
 ### 2. Dataset setup
 Download the dataset from [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
 
 Another dataset is customed [music_moods_dataset.csv](./music_moods_dataset.csv), you can find it in the file and upload it to colab for running
 
 

## Execution
### 1. Train Emotion Detection Models

### 2. Predict Emotion and Recommend Songs

### 3. Visualize Results 

#### Expected output
with taking a picture of a face,

then you can get a picture with emotional label and confidence and the 10 songs recomended to you.
 
![output_example](./image/output_example.png)



 ## Results
 
 ### 1. Emotion Detection
 Due to the imbalance among classes of the FER2013FER2013 dataset, we made data augmentation by using ImageDataGenerator to generate more examples of small-class images and assigning larger weights to underrepresented classes during training.
 Among fine-tuned CNN, ResNet and VGG16, CNN with hypeband parameter search have the best performance and generalization ability.
 
 Hyperparameter For CNN:
 
 ![Screenshot 2024-12-15 at 19 14 49](https://github.com/user-attachments/assets/a13509d1-51f7-4611-b595-930708cbc974)

 
 #### CNN Classification Report with 40 epochs


<img width="323" alt="MUSIC_CNN_model_classification_report" src="https://github.com/user-attachments/assets/7579baed-77dd-4b0a-a1bf-7bcd55140ee5" />
<img width="536" alt="cnn_confusion_matrix" src="https://github.com/user-attachments/assets/3def7bb8-b330-42a2-a5dc-09430cf9dfdc" />

 
 #### VGG-16 Classification Report with 40 epochs
<img width="312" alt="vgg_classification_report" src="https://github.com/user-attachments/assets/b2f59850-9d5e-4f15-bab2-c367f921f2dc" />


 ### 2.Music Recommendation
We initially implemented a CNN model for the music-emotion learning task and later enhanced it to an RNN with attention mechanism. This improvement yielded promising performance, significantly boosting the model's ability to detect smaller categories such as angry, fear, and surprise.
 ## Report for CNN
<img width="323" alt="MUSIC_CNN_model_classification_report" src="https://github.com/user-attachments/assets/c976b5b7-131b-405b-b3d0-3020852c0ff8" />

<img width="400" alt="MUSIC_CNN_model_confusion_matrix" src="https://github.com/user-attachments/assets/1fbb904c-d625-4ba6-b569-0b7fba533f04" />




 | ROC_curve_for_RNN_with_attention        | PR_curve_for_RNN_with_attention       |
|----------------|----------------|
| <img width="502" alt="MUSIC_RNN_ATTENTION_model_roc_curve" src="https://github.com/user-attachments/assets/5f551bd4-1579-477a-8890-d730b469d569" />| <img width="502" alt="MUSIC_RNN_ATTENTION_pr_curve" src="https://github.com/user-attachments/assets/8f7e59ff-1ca9-44ac-9dbc-b6557093640a" /> |





 ## Future Work


 ## Contributors
 Alexandra Przysucha (ajp9010@nyu.edu)
 Andrea Cardiel (alc9588@nyu.edu)
 Yuhong Zhang (yz9134@nyu.edu)
 
 

# DSUA-301-Final Project-Moodify
Most music recommendation systems today rely heavily on user history or listening habits, lacking real-time interaction based on users' current mood or context. 
This project combines real-time emotion detection with music recommendation to create a personalized and context-aware user experience.
We trained emotion detection models on the FER2013 dataset with fine-tuned CNN and VGG16 and paired them with a music recommendation model based on CNN and RNN-attention architectures.
The system can predict emotions like angry, sad, happy, surprise, and fear, then recommend 10 songs tailored to the detected emotion.

 ## Environment setup
 ### 1. requirements
 - Python >= 3.8
 - Install dependencies with pip:
   
 `pip install -r requirements.txt`

 
 ### 2. Dataset setup
 Download the dataset from [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
 
 Another dataset is customed [music_moods_dataset.csv](./music_moods_dataset.csv), you can find it in the file and upload it to Colab for running
 
 

## Execution
Try running the [Emotion_Music_Reccommendation.ipynb](./Emotion_Music_Reccommendation.ipynb) file in order.

#### Expected output
With taking a picture of a face (either input image as seen below or through the device's webcam feature,

then you can get a picture with emotional label and confidence and the 10 songs from different decades recommended to you.
 
<img width="500" alt="pascal_image_emotion_detection_with songs _cnn" src="https://github.com/user-attachments/assets/f4b073e7-d80e-4c2f-b69c-9b55c39bd36d" />

<br>

<img width="900" alt="youtube song recs for webcaem image detection happy" src="https://github.com/user-attachments/assets/ebd0dcf0-0a95-48d1-adf0-6a5876abaae4" />


 ## Results
 
 ### 1. Emotion Detection
 Due to the imbalance among classes of the FER2013FER2013 dataset, we made data augmentation by using ImageDataGenerator to generate more examples of small-class images and assigning larger weights to underrepresented classes during training.
 Among fine-tuned CNN and VGG16, CNN with Hyperband parameter search has the best performance and generalization ability.
 
 Hyperparameter For CNN:
 
 ![Screenshot 2024-12-15 at 19 14 49](https://github.com/user-attachments/assets/a13509d1-51f7-4611-b595-930708cbc974)

 
 #### CNN Classification Report with 40 epochs
<img width="400" alt="cnn_model_classification report" src="https://github.com/user-attachments/assets/1c58b7b7-72ad-47e7-ba40-883745501fb3" />

<br>
<img width="400" alt="cnn_confusion_matrix" src="https://github.com/user-attachments/assets/3def7bb8-b330-42a2-a5dc-09430cf9dfdc" />
<br>
 <img width="778" alt="cnn_model_accuracy_loss_plots" src="https://github.com/user-attachments/assets/6d9ed940-00d1-4f22-874a-8dcafdcc8047" />

 #### VGG-16 Classification Report with 40 epochs
<img width="400" alt="vgg_classification_report" src="https://github.com/user-attachments/assets/b2f59850-9d5e-4f15-bab2-c367f921f2dc" />
<br>
<img width="400" alt="vgg_confusion_matrix" src="https://github.com/user-attachments/assets/cb8b1267-dd74-4981-8696-79b54886e61d" />
<br>
<img width="782" alt="vgg_model_accuracy_and_loss_plots" src="https://github.com/user-attachments/assets/18b6807f-1343-4567-b033-d52062d4a0ff" />

The CNN model achieved an accuracy of approximately 66%, while the VGG-16 model
only reached around 48%. Despite applying class balancing techniques, both models
exhibited better recognition performance for major classes like happy compared to minor
10
classes such as fear. This result highlights the persistent challenge of class imbalance,
where underrepresented classes are more difficult to learn.
Both models were trained for 40 epochs, the CNN model demonstrated strong
generalization ability. Its training and validation accuracy curves align well, showing a
steady improvement over epochs. In contrast, the VGG-16 model encountered a
performance plateau for the validation set. A likely reason for this is that only the final
layers of the pre-trained VGG-16 model were unfrozen for fine-tuning, which constrained
its ability to adapt to the FER dataset’s specific characteristics.

#### Shifting from FER to FER+ dataset

![Screenshot 2024-12-15 at 16 22 53](https://github.com/user-attachments/assets/55c73c32-aa9f-4bb1-a3f4-1af989f52a9b)
<br>
![Screenshot 2024-12-15 at 16 23 21](https://github.com/user-attachments/assets/ea60e422-6e82-4f56-b42a-864d9686eb8d)

On FER+, the accuracy of the CNN model raises around 10% with 30 epochs, the
VGG-16 model raises around 5% with only 20 epochs. The imbalance of accuracy among
classes still exist as we are using the same image set, but the accuracy for all classes has
been improved and the gap between major class and minor class is shortened with the
multi-labeled annotation.
Our results demonstrate that, with identical data augmentation, models trained on the
FER+ dataset achieves significantly better performance. This finding highlights the
potential of our models when applied to more advanced and higher-quality datasets like
FER+.
 ### 2.Music Recommendation
We initially implemented a CNN model for the music-emotion learning task and later enhanced it to an RNN with an attention mechanism. This improvement yielded promising performance, significantly boosting the model's ability to detect smaller categories such as angry, fear, and surprise.
 ####  CNN Classification Report with 30 epochs
<img width="400" alt="MUSIC_CNN_model_classification_report" src="https://github.com/user-attachments/assets/c976b5b7-131b-405b-b3d0-3020852c0ff8" />
<br>
<img width="400" alt="MUSIC_CNN_model_confusion_matrix" src="https://github.com/user-attachments/assets/1fbb904c-d625-4ba6-b569-0b7fba533f04" />

For the music model, with 30 epochs, the CNN achieved 87% accuracy. However, there is a noticeable
gap between training and validation accuracy, indicating that the CNN model tends to
overfit after the initial epochs. The CNN may rely heavily on training patterns rather than
generalizing well to unseen data.

 ####  RNN+Attention Classification Report with 40 epochs

<img width="400" alt="MUSIC_RNN_ATTENTION_classification_report" src="https://github.com/user-attachments/assets/65b923ba-069e-4658-ab87-4144f6a0fddb" />

<br>
 | ROC_curve_for_RNN_with_attention               |                PR_curve_for_RNN_with_attention       |

| <img width="450" alt="MUSIC_RNN_ATTENTION_model_roc_curve" src="https://github.com/user-attachments/assets/5f551bd4-1579-477a-8890-d730b469d569" />| <img width="450" alt="MUSIC_RNN_ATTENTION_pr_curve" src="https://github.com/user-attachments/assets/8f7e59ff-1ca9-44ac-9dbc-b6557093640a" /> |

The RNN-attention model achieved a validation accuracy of approximately 90% after 40
epochs. The training and validation accuracy closely align, indicating minimal overfitting.
The convergence is smooth and stable after epoch 15, suggesting good learning
generalization.
Both models show strong performance for all classes but slightly reduced AUC for sad and
happy, suggesting challenges in capturing nuanced emotional features. Overall, the
Attention RNN Model outperforms the CNN Music Model due to better generalization,
stronger handling of class imbalance, and superior precision-recall performance across all
emotion classes.




 ## Future Work
 Shifting to more updated, higher-resolution datasets with additional dimensions of evaluating nuanced emotions.
 
 Find more precise or considerate music - mood mapping criteria.
 
 Integrating environmental context with emotional states and user historical preferences for music recommendations.
 

 ## Contributors
 Alexandra Przysucha (ajp9010@nyu.edu)
 Yuhong Zhang (yz9134@nyu.edu)
 Andrea Cardiel (alc9588@nyu.edu)

 
 

# 2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report(Jianghairi_2023324083)
# Final Report: Enhancing the Performance of Speech-To-Text (STT) Models in the Medical Field Using Non-Medical Field Data(11-A_group_task)

## 1. Introduction

Our research group focused on enhancing the performance of Speech-To-Text (STT) models in the medical field, with a specific emphasis on Korean senior speech datasets. This study addresses the growing need for AI in senior care and the challenges faced by medical AI, particularly the shortage of high-quality medical data and privacy concerns.

### 1.1 Background

- The global medical AI market is experiencing steady growth.
- There is an increasing need for senior care AI, with the elderly population expected to reach 20.6% by 2025.
- Major challenges in medical AI include:
  - Shortage of skilled AI experts
  - Lack of high-quality medical/healthcare data
  - Difficulties in accessing medical data due to privacy and security issues

### 1.2 Objective

Our primary objective was to develop a methodology to train STT models without relying heavily on medical data, thus addressing the challenges mentioned above.

## 2. Methodology

We employed the Whisper model, which is based on the Transformer architecture, for our speech-to-text tasks. Our approach involved three main strategies:

1. Establishing a benchmark
2. Data augmentation
3. Few-shot learning with Prototypical Network

### 2.1 Whisper Model Overview

The Whisper model utilizes the Transformer architecture, which includes:
- Self-attention and cross-attention mechanisms
- Positional encoding for sequence order representation

### 2.2 Benchmark Establishment

We fine-tuned the Whisper model using the following parameters:
- Dataset: Speech data from medical staff and patients for remote medical treatment
- Total data: 216,581 samples
- Train data: 10,000 samples
- Validation data: 5,000 samples
- Test data: 1,000 samples
- Hyperparameters:
  - Batch size: 4
  - Epochs: 20
  - Learning rate: 2e-5
  - Maximum length: 80

We generated benchmarks for the Tiny, Small, and Base versions of the Whisper model due to GPU limitations for larger models.

## 3. Data Augmentation

### 3.1 Methodology

Our data augmentation approach focused on enhancing the robustness and generalization capabilities of the STT model. We applied the following techniques:

1. Noise addition: Introducing controlled levels of background noise to simulate real-world conditions.
2. Reverberation: Adding echo effects to mimic different acoustic environments.
3. Distortion: Applying subtle distortions to the audio signal to increase variety.

### 3.2 Dataset

We used a dataset of 15,000 free talk samples, divided as follows:
- Train data: 12,000 samples
- Validation data: 1,500 samples
- Test data: 1,500 samples

The dataset was split into four equal parts, each receiving a different augmentation technique:
1. Noise addition
2. Reverberation
3. Distortion
4. No change (Raw audio)

### 3.3 Detail of data augmentation

data augmentation Code was uploaded in "audio_augmentation.ipynb"

### Data Augmentation

For audio processing, the following augmentation techniques were selected:

1. **Noise Augmentation:**
   ```python
   augment_noise = Compose([
       AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=1.0)
   ])
   ```
   * *Details:* Initially, the parameters for noise augmentation were set to `min_amplitude=0.01` and `max_amplitude=0.1`. However, this resulted in audio that was difficult to discern by the human ear. Additionally, small sample training of the STT model under these conditions yielded a character error rate (CER) of approximately 15-20%. Therefore, for the final model training, the amplitude of the noise was appropriately reduced to enhance performance.

2. **Reverb Augmentation:**
   ```python
   augment_reverb = Compose([
       TimeStretch(min_rate=0.95, max_rate=1.05, p=1.0)
   ])
   ```

3. **Distortion Augmentation:**
   ```python
   augment_distortion = Compose([
       PitchShift(min_semitones=-1, max_semitones=1, p=1.0)
   ])
   ```
   * *Details:* Both distortion and reverb augmentations underwent slight adjustments to optimize their impact on the final model training.

These augmentations were meticulously chosen and fine-tuned based on preliminary tests and observations to ensure an optimal balance between enhancing the training dataset and maintaining audio quality for human listeners. The adjustments led to improved performance and accuracy in the final speech-to-text model.


### 3.4 Results

We compared the performance of the augmented Whisper model against the base model trained on original samples. The results were as follows:

- Base model Character Error Rate (CER): 3.05
- Augmented model Character Error Rate (CER): 2.47

The noise-augmented Whisper model significantly outperformed the base model, demonstrating a notably lower CER. This improvement indicates that data augmentation techniques effectively enhanced the model's ability to handle various audio conditions and improved its overall accuracy in speech-to-text tasks.

### 3.4 Discussion

The success of data augmentation in improving the model's performance can be attributed to several factors:

1. Increased data diversity: By introducing various audio modifications, we expanded the range of input patterns the model could learn from, leading to better generalization.

2. Robustness to real-world conditions: The augmentation techniques simulated real-world audio challenges, helping the model become more resilient to noise, echoes, and distortions it might encounter in practical applications.

3. Overcoming data limitations: Given the scarcity of high-quality medical speech data, augmentation allowed us to artificially expand our dataset, mitigating some of the challenges associated with limited data availability.

4. Domain adaptation: Although we used non-medical data, the augmentation techniques may have helped bridge the gap between general speech patterns and the specific characteristics of medical conversations, particularly those involving senior speakers.

## 4. Few-Shot Learning with Prototypical Network

### 4.1 Methodology

To further improve our model's performance, we implemented few-shot learning using a Prototypical Network. This approach allowed us to leverage the knowledge of the pre-trained Whisper model and focus it on the medical domain using only a few labeled examples.

### 4.2 Mechanism

The few-shot learning process involved the following steps:
1. Whisper Feature Extraction
2. Embedding Layer
3. Defining Prototypical Loss
4. Training the Embedding Layer

The Prototypical Network aimed to represent each class with a prototype, calculated as the mean of the embeddings of the support examples in that class.

### 4.3 Results

We compared the performance of the few-shot learning model with the base zero-shot model:

- Zero-shot CER: 3.15
- Five-shot CER: 2.92
- Ten-shot CER: 2.84
- Fifteen-shot CER: 2.81

The few-shot models consistently outperformed the base model, demonstrating the effectiveness of this approach in adapting the pre-trained model to the medical domain.

## 5. Conclusions

Our research has successfully addressed key challenges in medical AI through the application of various techniques:

1. Fine-tuning the Whisper model improved overall performance.
2. Data augmentation techniques enhanced accuracy and model robustness.
3. Few-shot learning effectively leveraged limited data to adapt the model to the medical domain.

These approaches collectively demonstrated significant improvements in model accuracy, showcasing the potential of AI in healthcare applications, particularly for senior care.
## 6. Comparison with Project 11-b and Analysis of Methods**
In response to the valuable feedback received, we would like to provide additional insights into our results, particularly in comparison to Project 11-b, and elaborate on the differences in our methodologies.
### 6.1 Comparison of Results and Methodologies
While both our project and Project 11-b aimed to improve Speech-to-Text (STT) performance in the medical field, we adopted different approaches to achieve this goal.
Our Approach: Data Augmentation
Our method focused on data augmentation techniques applied directly to the audio data. The primary goal was to enhance the model's ability to perform well in challenging acoustic environments, such as:

Noisy conditions
Environments with significant background sounds
Situations with poor audio quality

By applying techniques like noise addition, reverberation, and distortion to our training data, we aimed to create a more robust model capable of handling real-world variability in audio inputs.
Project 11-b Approach: Ensemble Method
In contrast, Project 11-b utilized an ensemble method, combining different pre-trained models such as Whisper Base and Large, which had already undergone noise training. Their approach theoretically allows for greater model robustness by leveraging the strengths of multiple models.
## 6.2 Analysis of CER Improvements
we can discuss the improvements we observed in our Character Error Rate (CER) across different methods:

Baseline Whisper Model: CER of 3.05
Data Augmented Model: CER of 2.47
Few-shot Learning (15-shot): CER of 2.81

Our data augmentation technique showed a significant improvement, reducing the CER by 19% compared to the baseline. The few-shot learning approach also showed improvement, though not as substantial as the data augmentation method.
## 6.3 Comparative Analysis
The different approaches used by our project and Project 11-b each have their strengths:

Our Data Augmentation Approach:

Directly addresses variability in audio input quality
Potentially more efficient in terms of computational resources
Highly effective in improving performance on noisy or poor-quality audio


Project 11-b's Ensemble Approach:

Leverages the strengths of multiple pre-trained models
Potentially more robust across a wider range of scenarios
May achieve higher overall accuracy, especially on high-quality inputs



## 6.4 Dataset Considerations
It's important to note that the effectiveness of these methods can vary depending on the specific characteristics of the training and test datasets used.
Our training dataset consisted of 12,000 samples of free talk, augmented using various techniques. The test set comprised 1,500 samples, representing a mix of original and augmented data to simulate real-world conditions.
the ensemble method typically requires a substantial amount of diverse data to train effectively.
## 6.5 Future Directions
Given the strengths of both approaches, a promising direction for future research could be to combine data augmentation techniques with ensemble methods. This hybrid approach could potentially yield a model that is both robust to various audio conditions and highly accurate across a wide range of inputs.
Additionally, further investigation into the performance of both methods on specific subsets of medical speech data (e.g., different dialects, age groups, or medical specialties) could provide valuable insights for tailoring these approaches to specific use cases in the medical field.


##required data and a trained model
all used data and trained model was updated on **huggingface**
Dataset:
gingercake01/0607medical_data15000(including 15000 medical field audio data)
gingercake01/0604_15000_freetalk_4method(Contains 15,000 audios, divided into four equal parts, including noise augmentation, distortion, reverberation, and raw audio.)
gingercake01/15000free_talk_sample(including 15,000 senior citizens free conversation)

Model:
gingercake01/STT_15000_4method_audio_basev2_0607
The model gingercake01/STT_15000_4method_audio_basev2_0607 was trained using the dataset 0604_15000_freetalk_4method which underwent audio augmentation. This model achieved the highest accuracy in the transcription tasks in the medical domain.
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/dc87b9c6-53ea-41e2-a496-91862528024c)
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/e9d7c041-58f2-40e7-a8c0-6649f8720d75)


gingercake01/STT_15000audio_basev2_0606
The model STT_15000audio_basev2_0606 was trained using 15,000 freetalk datasets that were not subjected to audio augmentation, serving as a control model.
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/2aa0d038-6b19-493a-86e7-823bfa545a73)
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/a80fc0c2-d61b-417d-9ebc-d11cf6c9e6e7)

gingercake01/STT_1000audio_basev3
The model gingercake01/STT_1000audio_basev3 was initially trained with a small sample dataset to test the feasibility of the code and to prevent wasting GPU resources.
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/c6b97c67-6e9c-491c-99ce-6e636ce4e922)
![image](https://github.com/hairi0226/2024-MACHINE-LEARNING-AND-PROGRAMMING-final-report/assets/145079607/f93268fa-ff93-43a1-b604-53ca760ecc60)

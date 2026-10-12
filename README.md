
KidneyLife AI 

AI-Powered Kidney CT Image Classification

About the Project

KidneyLife AI is a deep learning application that classifies kidney CT images into four categories:

Normal
Cyst
Stone
Tumour

The system uses artificial intelligence to analyse medical images and provide a prediction with confidence scores. It aims to support healthcare professionals in identifying potential kidney conditions faster.

Note: KidneyLife AI supports clinical decision-making but does not replace professional medical diagnosis.

Problem Statement

South Africa faces a shortage of radiologists, which can contribute to delays in medical diagnosis. KidneyLife AI aims to help address this challenge by providing preliminary classification of kidney CT images.

Objectives
Classify kidney CT images into four categories.
Provide confidence scores for predictions.
Flag predictions below 80% confidence for manual review.
Support faster and more consistent image assessment.
Demonstrate the use of AI in healthcare.
Dataset

The project uses the CT Kidney Dataset on Kaggle.

The dataset contains 12,446 images:

Cyst: 3,709 images
Normal: 5,077 images
Stone: 1,377 images
Tumour: 2,283 images
Technologies Used
Python
TensorFlow and Keras
NumPy
Matplotlib
Scikit-learn
Git and GitHub
Visual Studio Code
Installation
Clone the repository:
   git clone <YOUR_REPOSITORY_URL>

Navigate to the project folder:
   cd <YOUR_PROJECT_DIRECTORY>

Install the required libraries:
   pip install tensorflow numpy matplotlib scikit-learn

Run the appropriate Python script to train or launch the application.

Note: Replace the placeholders and use the correct script name for your project.

Features
Kidney CT image classification.
Four prediction categories.
Confidence scores for each class.
Low-confidence prediction warnings.
Model performance evaluation.
Limitations
Supports kidney CT images only.
Limited to four classification categories.
The dataset was collected in Bangladesh, so performance on South African scans requires further validation.
Predictions may be incorrect and require professional review.
Future Improvements
Develop a user-friendly web interface.
Improve model accuracy and reliability.
Test with South African medical imaging data.
Integrate secure reporting features.
References

Islam, M.N. et al. (2022). Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography. Scientific Reports, 12(1).

Dataset: Kaggle – CT Kidney Dataset

Disclaimer

KidneyLife AI is a research and decision-support project. It is not a replacement for qualified healthcare professionals and must not be used as the sole basis for medical diagnosis or treatment.

# Spam-email-Classification 
📄 Title Page
Problem Title: Classify Emails as Spam or Not Spam Using Structured Metadata
Name: [Your Name]
Roll Number: [Your Roll No]
Course: AI - Mid-Semester Exam
Tool Used: Google Colab
Dataset: spam_emails.csv

🔍 Introduction
Email spam detection is a critical task in digital communication systems. In this problem, we aim to classify emails into "Spam" or "Not Spam" based on metadata such as the number of links, attachments, and the reputation of the sender. This is a binary classification task that helps in preventing unwanted emails and phishing attacks.

We use a machine learning model trained on a labeled dataset to make predictions about whether an email is spam or not. Below is a snapshot of what the data looks like:

num_links: Number of hyperlinks in the email

num_attachments: Number of attached files

sender_reputation: A numeric score representing the trustworthiness of the sender

is_spam: Label - Yes (spam) / No (not spam)

⚙️ Methodology
Data Preprocessing:

The target variable is_spam is encoded to 0 (No) and 1 (Yes).

Features are selected from structured metadata.

Model Used:

A Random Forest Classifier is used due to its robustness and ability to handle nonlinear features well.

Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix Visualization

📚 References/Credits
Dataset: Provided (spam_emails.csv)

Libraries: pandas, sklearn, matplotlib, seaborn

Platform: Google Colab

Algorithm: Random Forest Classifier (scikit-learn)

Guidance and instruction from course materials and faculty

# Fraudulent-Job-Posting-Statistical-Model

This numerical simulation of fraudelent vs. non-fraudulent job postings using Naive Bayes probabalistic model. These files investigate different parameters within the given CSV file, including, location, has company logo, has company profile, has salary range, has screening questions, and text in the job postings. Some of simulations create heat maps for the boolean values to find whether present parameters indicate whether a job is fraudulent or not. The other simulations include word association witht the likelihood of certain words that appear in a job posting indicting the likelihood that a job was fake. This repository includes a data folder and scripts for each of the simulations. The scripts are split into parameters: has company logo and has copmany profile, has screening questions and has salary range, location, and word association for job descriptions. 

Running the files:

location.m, profile_logo.m, and salary_questions.m:
To run location.m, profile_logo.m, and salary_questions.m, nothing needs to be changed about these files:
Steps:
1. Download the repository and extract the files from the zip file
2. Open MatLab
3. Open the folder where you extracted the files into
4. Open location.m, profile_logo.m, or salary_questions.m and press run

predict_NaiveBayes.m, predict_naiveBayesTwoPass.m, run_NaiveBayes.m, and run_NaiveBayesTwoPass.m:
To run predict_NaiveBayes.m, predict_naiveBayesTwoPass.m, run_NaiveBayes.m, and run_NaiveBayesTwoPass.m, you may need to download certain toolboxes: Text Analystics Toolbox.
The predict_{file}.m files will predict the likelihood of a job posting to be fake. You can analyze different job postings by inserting a different text file in this line: 
newPost = fileread("./data/fake_AIWithTitle.txt");
The run_{file}.m files analyze the given dataset and output how likely a job is to be fraudulent if certain words appears either in the job title, job description, or in the benfits section wihtin the posting. 
The {file}TwoPass.m files train the model twice, increasing accuracy of the model. 
Steps:
1. Run run_NaiveBayes.m or run_NaiveBayesTwoPass.m (depends on which model you prefer).
2. If you want to predict your own job posting, replace newPost = fileread("./data/fake_AIWithTitle.txt"); with your text file (assuming it is already placed within the folder with all these files).
3. Run predict_NaiveBayes.m or predict_naiveBayesTwoPass.m depending on which model you prefer with the next text file inserted. 


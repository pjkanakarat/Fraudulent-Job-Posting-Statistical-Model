# Fraudulent-Job-Posting-Statistical-Model

This is a numerical simulation of fraudelent vs. non-fraudulent job postings using Naive Bayes probabalistic model. These files investigate different parameters within the given datset, including location, has company logo, has company profile, has salary range, has screening questions, and text within the job postings. Some of simulations create heat maps to find whether present parameters indicate if a job is fraudulent or not fraudulent. The other simulations are word analyses of the job postings within the dataset: certain words that appear in the job title, job description, and job benefits indicate the probability that a job posting is fake. There are also a couple scripts that use this word model to find the probabaility of a specific job posting to be fraudulent. This repository includes a data folder and scripts for each of the simulations. The scripts are split into parameters: has company logo and has company profile, has screening questions and has salary range, location, word analysis for job postings, and indivudual job posting fraudelence probability. 

Running the files:

location.m, profile_logo.m, and salary_questions.m:
To run location.m, profile_logo.m, and salary_questions.m, nothing needs to be changed about these files:

Steps:
1. Download the repository and extract the files from the zip file
2. Open MatLab
3. Open the folder where you placed the extracted the files
4. Open location.m, profile_logo.m, or salary_questions.m
5. Press run
   
predict_NaiveBayes.m, predict_naiveBayesTwoPass.m, run_NaiveBayes.m, and run_NaiveBayesTwoPass.m:
To run predict_NaiveBayes.m, predict_naiveBayesTwoPass.m, run_NaiveBayes.m, and run_NaiveBayesTwoPass.m, you may need to download certain toolboxes: Text Analystics Toolbox.
The predict_{file}.m files will predict the probability of a job posting to be fake. You can analyze different job postings by inserting a different text file in this line: 
newPost = fileread("./data/fake_AIWithTitle.txt");
The run_{file}.m files analyze the given dataset and output how likely a job is to be fraudulent if certain words appears either in the job title, job description, or in the benefits section within the posting. 
The {file}TwoPass.m files train the model twice, increasing accuracy of the model. 

Steps:
1. Download the repository and extract the files from the zip file
2. Open Matlab
3. Open the folder where you placed the extracted the files
4. Open run_NaiveBayes.m or run_NaiveBayesTwoPass.m (depends on which model you prefer)
5. Press Run
6. If you want to predict your own job posting, open predict_NaiveBayes.m or predict_naiveBayesTwoPass.m (depending on which model you chose before)
7. replace newPost = fileread("./data/fake_AIWithTitle.txt") in the file with your text file of the job posting you want to analyze (assuming the text file is already placed within the folder with all of these files)
8. Press run


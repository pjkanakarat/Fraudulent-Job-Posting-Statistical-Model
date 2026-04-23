%% Naive Bayes vs Joint Distribution Visualization

clc, clear

% load csv
data = readtable("./data/fake_job_postings.csv");

for i = 1:height(data)
    if ~isempty(data.salary_range{i}) && ~strcmp(data.salary_range{i}, '')
        data.has_salary_range(i) = 1;
    else
        data.has_salary_range(i) = 0;
    end
end

% load variables
isFraud = data.fraudulent;

%% Fraudelent Job Postings
fraud_idx = isFraud == 1;

salary_range_f = data.has_salary_range(fraud_idx);
questions_f = data.has_questions(fraud_idx);

% compute Naive Bayes
p_salary_range = zeros(1,2);
p_questions = zeros(1,2);

for i = 0:1
    p_salary_range(i+1) = sum(salary_range_f == i) / length(salary_range_f);
    p_questions(i+1) = sum(questions_f == i) / length(questions_f);
end

% joint Naive Bayes
joint_nb = p_salary_range' * p_questions;

% plot
figure
heatmap({'No Questions','Questions'}, {'No Salary Range','Salary Range'}, joint_nb)
title('Naive Bayes Approximation for Fraudulent')
disp('Naive Bayes Approximation:')
disp(joint_nb)

% repeat with not fraudelent
fraud_nidx = isFraud == 0;

salary_range_nf = data.has_salary_range(fraud_nidx);
questions_nf = data.has_questions(fraud_nidx);


% Naive Bayes
p_nsalary_range = zeros(1,2);
p_nquestions = zeros(1,2);

for i = 0:1
    p_nsalary_range(i+1) = sum(salary_range_nf == i) / length(salary_range_nf);
    p_nquestions(i+1) = sum(questions_nf == i) / length(questions_nf);
end

% joint Naive Bayes
joint_nnb = p_nsalary_range' * p_nquestions;

% plot
figure
heatmap({'No Questions','Questions'}, {'No Salary Range','Salary Range'}, joint_nnb)
title('Naive Bayes Approximation For Not Fraudulent')
disp('Naive Bayes Approximation:')
disp(joint_nnb)


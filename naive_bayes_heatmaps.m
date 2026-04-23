%% Naive Bayes vs Joint Distribution Visualization

clc, clear

% load csv
data = readtable("./data/fake_job_postings.csv");

for i = 1:height(data)
    if ~isempty(data.company_profile{i}) && ~strcmp(data.company_profile{i}, '')
        data.has_company_profile(i) = 1;
    else
        data.has_company_profile(i) = 0;
    end
end

% load variables
isFraud = data.fraudulent;
hasProfile = data.has_company_profile;
hasLogo = data.has_company_logo;

%% Fraudelent Job Postings
fraud_idx = isFraud == 1;

profile_f = hasProfile(fraud_idx);
logo_f = hasLogo(fraud_idx);


% compute Naive Bayes
p_profile = zeros(1,2);
p_logo = zeros(1,2);

for i = 0:1
    p_profile(i+1) = sum(profile_f == i) / length(profile_f);
    p_logo(i+1) = sum(logo_f == i) / length(logo_f);
end

% joint Naive Bayes
joint_nb = p_profile' * p_logo;

% plot
figure
heatmap({'No Logo','Logo'}, {'No Profile','Profile'}, joint_nb)
title('Naive Bayes Approximation for Fraudulent')
disp('Naive Bayes Approximation:')
disp(joint_nb)

% repeat with not fraudelent
fraud_nidx = isFraud == 0;

profile_nf = hasProfile(fraud_nidx);
logo_nf = hasLogo(fraud_nidx);


% Naive Bayes
p_nprofile = zeros(1,2);
p_nlogo = zeros(1,2);

for i = 0:1
    p_nprofile(i+1) = sum(profile_nf == i) / length(profile_nf);
    p_nlogo(i+1) = sum(logo_nf == i) / length(logo_nf);
end

% joint Naive Bayes
joint_nnb = p_nprofile' * p_nlogo;

% plot
figure
heatmap({'No Logo','Logo'}, {'No Profile','Profile'}, joint_nnb)
title('Naive Bayes Approximation For Not Fraudulent')
disp('Naive Bayes Approximation:')
disp(joint_nnb)


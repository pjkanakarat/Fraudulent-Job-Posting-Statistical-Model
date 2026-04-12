clc, clear

% load csv
data = readtable("fake_job_postings.csv");

for i = 1:height(data)
    if ~isempty(data.company_profile{i}) && ~strcmp(data.company_profile{i}, '')
        data.has_company_profile(i) = 1;
    else
        data.has_copmany_profile(i) = 0;
    end
end

% make variables for the columns in the table
isFraud = data.fraudulent;
hasProfile = data.has_company_profile;
hasLogo = data.has_company_logo;

% fraud
fraud_idx_1 = isFraud == 1;

profile_f_1 = hasProfile(fraud_idx_1);
logo_f_1 = hasLogo(fraud_idx_1);

% not fraud
fraud_idx_0 = isFraud == 0;

profile_f_0 = hasProfile(fraud_idx_0);
logo_f_0 = hasLogo(fraud_idx_0);

% joint counts
joint_counts_1 = zeros(2,2);

for i = 0:1
    for j = 0:1
        joint_counts_1(i+1, j+1) = sum(profile_f_1 == i & logo_f_1 == j);
    end
end

% Convert to probabilities
joint_probs_1 = joint_counts_1 / sum(joint_counts_1(:));

figure
heatmap({'No Logo','Logo'}, {'No Profile','Profile'}, joint_probs_1)
title('Joint Probability for Fraudulent Postings')
xlabel('Logo')
ylabel('Profile')

% joint counts
joint_counts_0 = zeros(2,2);

for i = 0:1
    for j = 0:1
        joint_counts_0(i+1, j+1) = sum(profile_f_0 == i & logo_f_0 == j);
    end
end

% Convert to probabilities
joint_probs_0 = joint_counts_0 / sum(joint_counts_0(:));

figure
heatmap({'No Logo','Logo'}, {'No Profile','Profile'}, joint_probs_0)
title('Joint Probability for Not Fraudulent Postings')
xlabel('Logo')
ylabel('Profile')
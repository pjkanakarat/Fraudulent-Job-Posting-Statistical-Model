%% Step 5: Validate on "Fake" Data
% 1. Load the synthetic data
fakeData = readtable('synthetic_job_data_v2.csv');

% 2. Merge columns as per your original model
rawTextFake = fakeData{:, 7} + " " + fakeData{:, 8} + " " + fakeData{:, 9};
labelsFake = fakeData.fraudulent;

% 3. Preprocess and Bag-of-Words
documentsFake = tokenizedDocument(rawTextFake);
documentsFake = lower(documentsFake);
documentsFake = erasePunctuation(documentsFake);
documentsFake = removeStopWords(documentsFake);
documentsFake = normalizeWords(documentsFake); % Lemmatization (e.g., "runs" -> "run")

bagFake = bagOfWords(documentsFake);
XFake = full(bagFake.Counts);

%% Step 4: Statistical Validation (Mean and Variance)
% We programmed the Python script to include 15 words per post.
% Therefore: Expected Mean = 15 * Probability

% 1. Identify indices for 'wire' and 'experience' in the vocabulary
idxWire = find(wordsFake == "wire");
idxExp  = find(wordsFake == "experi"); % Note: Stemmed form

% 2. Calculate Empirical Mean and Variance from the CSV data
% For Fraudulent Postings (Label = 1)
fraudCounts = XFake(labelsFake == 1, :);
meanWireFraud = mean(fraudCounts(:, idxWire));
varWireFraud  = var(fraudCounts(:, idxWire));

% For Legitimate Postings (Label = 0)
legitCounts = XFake(labelsFake == 0, :);
meanExpLegit = mean(legitCounts(:, idxExp));
varExpLegit  = var(legitCounts(:, idxExp));

% 3. Compare to Theoretical Targets
fprintf('\n--- Step 4: Statistical Validation ---\n');
fprintf('WORD: "wire" (Fraud Class)\n');
fprintf('  Target Mean: 6.00 (15 * 0.40)\n'); 
fprintf('  Measured Mean: %.2f\n', meanWireFraud);
fprintf('  Measured Var:  %.2f (Target: 3.60)\n', varWireFraud);

fprintf('\nWORD: "experience" (Legit Class)\n');
fprintf('  Target Mean: 6.00 (15 * 0.40)\n');
fprintf('  Measured Mean: %.2f\n', meanExpLegit);
fprintf('  Measured Var:  %.2f (Target: 3.60)\n', varExpLegit);

% 4. Formal Statistical Test (One-Sample t-test)
% Does the data come from a distribution with the expected mean?
[h, p] = ttest(fraudCounts(:, idxWire), 6.00);
if h == 0
    fprintf('\nt-test Result: PASS (p = %.4f). The simulation matches the model.\n', p);
else
    fprintf('\nt-test Result: FAIL. The data deviates from the model.\n');
end

% 4. Train the Model
MdlFake = fitcnb(XFake, labelsFake, ...
    'DistributionNames', 'mn', ...
    'Prior', [0.8, 0.2]);

% 5. Extract Fraud Association Scores
wordsFake = bagFake.Vocabulary;
probNotFraudFake = cell2mat(MdlFake.DistributionParameters(1, :));
probFraudFake = cell2mat(MdlFake.DistributionParameters(2, :));
fraudRatioFake = probFraudFake ./ (probNotFraudFake + eps);

% Create Table
validationTable = table(wordsFake(:), fraudRatioFake(:), ...
    'VariableNames', {'Word', 'Fraud_Score'});
validationTable = sortrows(validationTable, 'Fraud_Score', 'descend');

% 6. Interpretation
fprintf('Validation Results:\n');
disp(head(validationTable, 10));

% CHECK: Does 'Wire' have a high score? 
% In the simulation, Wire is ~60x more likely in fraud. 
% If your code shows a high score for 'Wire', Step 5 is validated.
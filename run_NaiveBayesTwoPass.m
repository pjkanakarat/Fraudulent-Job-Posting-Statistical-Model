%% 1.Load Data
data = readtable('./data/fake_job_postings.csv');

% 7: Description, 8: Requirements, 9: Benefits, and column: Fraudulent
rawText = data{:, 7} + " " + data{:, 8} + " " + data{:, 9};
labels = data.fraudulent; % Categorical or numeric (0/1)

%% 2. Text Cleaning
text = lower(rawText);
text = eraseURLs(text);
text = erasePunctuation(text);


documents = tokenizedDocument(text); %Not cleaning
documents = removeStopWords(documents);
documents = removeLongWords(documents, 15);

%% 3. Create Bag-of-Words
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag, 5); %Removes words appearing in <5 entries

% Convert bag into matrix
X = full(bag.Counts); 
words_initial = bag.Vocabulary;

%% 4. Pass 1: Train Model and Identify Noise Words
% Train a temporary model to find the probability floor
priorDist = [0.50, 0.50]; 
tempMdl = fitcnb(X, labels, ...
    'DistributionNames', 'mn', ...
    'Prior', priorDist);

%automatically gets legit class (0 or false) by default
probLegit = cell2mat(tempMdl.DistributionParameters(1, :));

% % Handle MATLAB's [mean, std] storage if it occurred
% if length(probLegit) > length(words_initial)
%     probLegit = probLegit(1:2:end); 
% end

%% 5. Feature Selection: Filter out the "Zero-Legit" words
goodWordIdx = find(probLegit > 3.621e-07); %slightly above 3.6204e-07

% Filter the data matrix and vocabulary list using logical indexing
X_final = X(:, goodWordIdx);
words_final = words_initial(goodWordIdx);

fprintf('Filtered: Removed %d words.', length(words_initial) - length(goodWordIdx));

%% 6. Final Model Training
% This is the model you will use for your actual predictions
Mdl = fitcnb(X_final, labels, ...
    'DistributionNames', 'mn', ...
    'Prior', priorDist, ...
    'ClassNames', [0, 1]);

%% 7. Word Association Analysis (Filtered Results)
probLegitFinal = cell2mat(Mdl.DistributionParameters(1, :));
probFraudFinal = cell2mat(Mdl.DistributionParameters(2, :));

if length(probLegitFinal) > length(words_final)
    probLegitFinal = probLegitFinal(1:2:end);
    probFraudFinal = probFraudFinal(1:2:end);
end

fraudRatio = probFraudFinal ./ (probLegitFinal + eps);

wordAssociations = table(words_final(:), ...
                         probLegitFinal(:), ...
                         probFraudFinal(:), ...
                         fraudRatio(:), ...
    'VariableNames', {'Word', 'Prob_Legit', 'Prob_Fraud', 'Fraud_Ratio'});

wordAssociations = sortrows(wordAssociations, 'Fraud_Ratio', 'descend');

fprintf('\nTop 20 words found in both classes):\n');
disp(head(wordAssociations, 20));

%% 8. Save for Prediction Scripts
save('NB_Model_Data.mat', 'Mdl', 'bag', 'goodWordIdx');
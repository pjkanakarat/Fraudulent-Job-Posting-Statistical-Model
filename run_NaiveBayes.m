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
documents = normalizeWords(documents); %stemming | REMOVE TO GET ORIGINAL DATA

%% 3. Create Bag-of-Words
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag, 5); %Removes words appearing in <5 entries

% Convert bag into matrix
X = full(bag.Counts); 
words = bag.Vocabulary;

%% 4. Train Naive Bayes Model
% We specify 'Multinomial' because we are dealing with word counts
% We set the Prior: [Non-Fraud (0), Fraud (1)] = [0.80, 0.20]
priorDist = [0.50, 0.50]; 

Mdl = fitcnb(X, labels, ...
'DistributionNames', 'mn', ...
    'Prior', priorDist, ...
    'ClassNames', [0, 1]);

%% 5. Word Association Analysis
% 'cell2mat' collects the values from every cell in the row (all words)
% We use (1, :) to say "all columns for class 1"
probNotFraud = cell2mat(Mdl.DistributionParameters(1, :)); 
probFraud    = cell2mat(Mdl.DistributionParameters(2, :));

% Calculate the ratio
fraudRatio = probFraud ./ (probNotFraud + eps);

% Create the table using (:) to force everything into column vectors
wordAssociations = table(words(:), ...
                         probNotFraud(:), ...
                         probFraud(:), ...
                         fraudRatio(:), ...
    'VariableNames', {'Word', 'Prob_Legit', 'Prob_Fraud', 'Fraud_Ratio'});

% Sort and display
wordAssociations = sortrows(wordAssociations, 'Fraud_Ratio', 'descend');
fprintf('Top Words Associated with Fraudulent Postings:\n');
disp(head(wordAssociations, 20));
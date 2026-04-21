%% 1. Load the Data
% Assuming the file is named 'job_postings.csv'
data = readtable('synthetic_job_data_v2.csv');

% Extract the relevant columns (7: Description, 8: Requirements, 9: Benefits)
% and the label (assuming it is named 'fraudulent')
rawText = data{:, 7} + " " + data{:, 8} + " " + data{:, 9};
labels = data.fraudulent; % Categorical or numeric (0/1)

%% 2. Text Preprocessing
% Convert to tokenized documents
documents = tokenizedDocument(rawText);

% Cleaning: lowercase, remove punctuation, and remove common stop words
documents = lower(documents);
documents = erasePunctuation(documents);
documents = removeStopWords(documents);
documents = normalizeWords(documents); % Lemmatization (e.g., "runs" -> "run")

%% 3. Create Bag-of-Words
% Create the bag and remove very rare words to reduce noise
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag, 5); % Removes words appearing in < 5 posts

% Convert the bag into a numeric matrix (word counts per document)
X = full(bag.Counts); 
words = bag.Vocabulary;

%% 4. Train Naive Bayes Model
% We specify 'Multinomial' because we are dealing with word counts
% We set the Prior: [Non-Fraud (0), Fraud (1)] = [0.80, 0.20]
priorDist = [0.80, 0.20]; 

Mdl = fitcnb(X, labels, ...
    'DistributionNames', 'mn', ...
    'Prior', priorDist, ...
    'ClassNames', [0, 1]);

%% 5. Word Association Analysis (Robust Version)
% 'cell2mat' collects the values from every cell in the row (all words)
% We use (1, :) to say "all columns for class 1"
probNotFraud = cell2mat(Mdl.DistributionParameters(1, :)); 
probFraud    = cell2mat(Mdl.DistributionParameters(2, :));

% Check: If the model is Multinomial, it sometimes stores [mean, std] 
% per cell if it defaulted to Normal. Let's ensure we have the right length.
% If size is 1 x (2*15459), we just need the first half (the means).
if length(probNotFraud) > length(words)
    probNotFraud = probNotFraud(1:2:end); % Get every 1st element (means)
    probFraud    = probFraud(1:2:end);
end

% Calculate the ratio (Fraud vs. Legit)
fraudRatio = probFraud ./ (probNotFraud + eps);

% Create the table using (:) to force everything into column vectors
wordAssociations = table(words(:), ...
                         probNotFraud(:), ...
                         probFraud(:), ...
                         fraudRatio(:), ...
    'VariableNames', {'Word', 'Prob_Legit', 'Prob_Fraud', 'Fraud_Association_Score'});

% Sort and display top 20
wordAssociations = sortrows(wordAssociations, 'Fraud_Association_Score', 'descend');
fprintf('Top 20 Words Associated with Fraudulent Postings:\n');
disp(head(wordAssociations, 20));

%% 6. (Optional) Example Prediction
% newPost = "Urgent hire for data entry. No experience needed. High pay.";
% newDoc = tokenizedDocument(newPost);
% newX = encode(bag, newDoc);
% prediction = predict(Mdl, newX);
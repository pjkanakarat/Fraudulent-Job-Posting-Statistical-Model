%Load model and the filter index
load('NB_Model_Data.mat'); 

%Clean
newPost = fileread("./data/fake_AIWithTitle.txt");
text = lower(newPost);
text = eraseURLs(text);
text = erasePunctuation(text);

documents = tokenizedDocument(text); %Not cleaning
documents = removeStopWords(documents);
documents = removeLongWords(documents, 15);

newX_full = encode(bag, documents);

newX_filtered = newX_full(goodWordIdx);

% 5. Predict
[label, posteriors] = predict(Mdl, full(newX_filtered));
fprintf('Prediction: %d (1=Fraud, 0=Legit)\n', label);
fprintf('Probability of Fraud: %.2f%%\n', posteriors(2) * 100);
%Read job posting
rawText = fileread("./data/real_Entry.txt");

%Clean
text = lower(rawText);
text = eraseURLs(text);
text = erasePunctuation(text);

documents = tokenizedDocument(text);
documents = removeStopWords(documents);
documents = removeLongWords(documents, 15);


newX = encode(bag, documents);
%Predict the label and the probability
[label, posteriors] = predict(Mdl, full(newX));

fprintf('Prediction: %d (1=Fraud, 0=Legit)\n', label);
fprintf('Probability of Fraud: %.2f%%\n', posteriors(2) * 100);
import sys
from collections import defaultdict
from timeit import default_timer as timer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import warnings

from FinalClassification import load_data, classifier_based_on_number, check_neutral_existence

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# do the classification with printing the results on a file
def SA_TFIDF(dataset_number, stemmer_number, classifier_number):
    output = load_data(dataset_number, stemmer_number)
    # get the output from applying pre processing for the data
    # and save them into x_train x_test and y_train y_test

    if dataset_number == 7:
        pos_train_data_1, pos_train_labels_1 = output[0]['output']
        neg_train_data_1, neg_train_labels_1 = output[1]['output']
        neu_train_data_1, neu_train_labels_1 = output[2]['output']
        pos_test_data_1, pos_test_labels_1 = output[3]['output']
        neg_test_data_1, neg_test_labels_1 = output[4]['output']
        neu_test_data_1, neu_test_labels_1 = output[5]['output']

        pos_train_data_2, pos_train_labels_2 = output[6]['output']
        neg_train_data_2, neg_train_labels_2 = output[7]['output']
        neu_train_data_2, neu_train_labels_2 = output[8]['output']
        pos_test_data_2, pos_test_labels_2 = output[9]['output']
        neg_test_data_2, neg_test_labels_2 = output[10]['output']
        neu_test_data_2, neu_test_labels_2 = output[11]['output']

        pos_train_data_3, pos_train_labels_3 = output[12]['output']
        neg_train_data_3, neg_train_labels_3 = output[13]['output']
        neu_train_data_3, neu_train_labels_3 = output[14]['output']
        pos_test_data_3, pos_test_labels_3 = output[15]['output']
        neg_test_data_3, neg_test_labels_3 = output[16]['output']
        neu_test_data_3, neu_test_labels_3 = output[17]['output']

        pos_train_data_4, pos_train_labels_4 = output[18]['output']
        neg_train_data_4, neg_train_labels_4 = output[19]['output']
        pos_test_data_4, pos_test_labels_4 = output[20]['output']
        neg_test_data_4, neg_test_labels_4 = output[21]['output']

        pos_train_data_5, pos_train_labels_5 = output[22]['output']
        neg_train_data_5, neg_train_labels_5 = output[23]['output']
        neu_train_data_5, neu_train_labels_5 = output[24]['output']
        pos_test_data_5, pos_test_labels_5 = output[25]['output']
        neg_test_data_5, neg_test_labels_5 = output[26]['output']
        neu_test_data_5, neu_test_labels_5 = output[27]['output']

        pos_train_data_6, pos_train_labels_6 = output[28]['output']
        neg_train_data_6, neg_train_labels_6 = output[29]['output']
        neu_train_data_6, neu_train_labels_6 = output[30]['output']
        pos_test_data_6, pos_test_labels_6 = output[31]['output']
        neg_test_data_6, neg_test_labels_6 = output[32]['output']
        neu_test_data_6, neu_test_labels_6 = output[33]['output']

        pos_train = pos_train_data_1 + pos_train_data_2 + pos_train_data_3 + pos_train_data_5 + pos_train_data_6 + pos_train_data_4
        neg_train = neg_train_data_1 + neg_train_data_2 + neg_train_data_3 + neg_train_data_5 + neg_train_data_6 + neg_train_data_4
        neu_train = neu_train_data_1 + neu_train_data_2 + neu_train_data_3 + neu_train_data_5 + neu_train_data_6

        pos_train_labels = pos_train_labels_1 + pos_train_labels_2 + pos_train_labels_3 + pos_train_labels_5 + pos_train_labels_6 + pos_train_labels_4
        neg_train_labels = neg_train_labels_1 + neg_train_labels_2 + neg_train_labels_3 + neg_train_labels_5 + neg_train_labels_6 + neg_train_labels_4
        neu_train_labels = neu_train_labels_1 + neu_train_labels_2 + neu_train_labels_3 + neu_train_labels_5 + neu_train_labels_6

        pos_test = pos_test_data_1 + pos_test_data_2 + pos_test_data_3 + pos_test_data_5 + pos_test_data_6 + pos_test_data_4
        neg_test = neg_test_data_1 + neg_test_data_2 + neg_test_data_3 + neg_test_data_5 + neg_test_data_6 + neg_test_data_4
        neu_test = neu_test_data_1 + neu_test_data_2 + neu_test_data_3 + neu_test_data_5 + neu_test_data_6

        pos_test_labels = pos_test_labels_1 + pos_test_labels_2 + pos_test_labels_3 + pos_test_labels_5 + pos_test_labels_6 + pos_test_labels_4
        neg_test_labels = neg_test_labels_1 + neg_test_labels_2 + neg_test_labels_3 + neg_test_labels_5 + neg_test_labels_6 + neg_test_labels_4
        neu_test_labels = neu_test_labels_1 + neu_test_labels_2 + neu_test_labels_3 + neu_test_labels_5 + neu_test_labels_6

        x_train = pos_train + neg_train + neu_train

        y_train = pos_train_labels + neg_train_labels + neu_train_labels

        x_test = pos_test + neg_test + neu_test

        y_test = pos_test_labels + neg_test_labels + neu_test_labels
    else:
        if check_neutral_existence:
            pos_train_data, pos_train_labels = output[0]['output']
            neg_train_data, neg_train_labels = output[1]['output']
            neu_train_data, neu_train_labels = output[2]['output']
            pos_test_data, pos_test_labels = output[3]['output']
            neg_test_data, neg_test_labels = output[4]['output']
            neu_test_data, neu_test_labels = output[5]['output']

            x_train = pos_train_data + neg_train_data + neu_train_data

            y_train = pos_train_labels + neg_train_labels + neu_train_labels

            x_test = pos_test_data + neg_test_data + neu_test_data

            y_test = pos_test_labels + neg_test_labels + neu_test_labels

        else:
            pos_train_data, pos_train_labels = output[0]['output']
            neg_train_data, neg_train_labels = output[1]['output']
            pos_test_data, pos_test_labels = output[2]['output']
            neg_test_data, neg_test_labels = output[3]['output']

            x_train = pos_train_data + neg_train_data
            y_train = pos_train_labels + neg_train_labels

            x_test = pos_test_data + neg_test_data

            y_test = pos_test_labels + neg_test_labels

    # printing data info
    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: # of pos:{}\t# of neg:{}\t'.format(y_train.count('P'), y_train.count('N')))
    print('test data: # of pos:{}\t# of neg:{}\t'.format(y_test.count('P'), y_test.count('N')))
    print('------------------------------------')

    # get the classifier with the best parameter for the knn, DT, RFT classifiers
    clf = classifier_based_on_number(classifier_number, x_train, y_train, x_test, y_test, dataset_number)

    # load the classifier in the pipeline
    pipeline = Pipeline([
        ('main_vect', TfidfVectorizer(
            analyzer='word', lowercase=False,
            ngram_range=(1, 2)
        )),
        ('main_clf', clf),
    ])

    # train the model which train it after applying TF-IDF on it using the pipeline automatically
    pipeline.fit(x_train, y_train)

    # get the features
    feature_names = pipeline.named_steps['main_vect'].get_feature_names()

    # print the number of features extracted
    print('features:', )
    print(len(feature_names), 'are kept')
    print('features are selected')

    # test the model which test it after applying TF-IDF on it using the pipeline automatically
    y_predicted = pipeline.predict(x_test)

    # printing the classification report for the classifier
    if check_neutral_existence:
        # Print the classification report
        print(metrics.classification_report(y_test, y_predicted,
                                            target_names=['P', 'N', 'U']))
    else:
        # Print the classification report
        print(metrics.classification_report(y_test, y_predicted,
                                            target_names=['P', 'N']))

    # Print the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    print('# of features:', len(feature_names))

    # printing first 100 tweet for observation
    for i in range(0, 25):
        print(x_test[i] + '\t' + 'Sentiment: ' + y_predicted[i])


# save the results of the classifier in a file in the results folder
def save_sa_results(dataset, stemmer, classifier):
    # the file name depending on the n-gram number with the dataset, stemmer and classifier number
    outfile = 'results/Classification using TF-IDF for ' + str(dataset) + '_' + str(stemmer) + '_' + str(
        classifier) + '.result'
    # open the file we want to write on
    sys.stdout = open(outfile, mode='w', encoding='utf-8')
    # timer for calculate the time needed to do the classification
    start = timer()
    print('classification using TF-IDF for ' + str(dataset) + '_' + str(stemmer) + '_' + str(classifier))
    # do the classification
    # each print after we opened the file is executed on the file not on the console
    SA_TFIDF(dataset, stemmer, classifier)
    end = timer()
    # we print the time took for the classifier
    print('time taking for training the classifier: ' + str(end - start))
    sys.stdout.close()


# when we need to classify we need to have it in the main
# because we used parallel preprocessing for the data
if __name__ == '__main__':
    # we choose the desired dataset, stemmer and classifier number
    # we want to build the model on it.
    dataset = 9
    classifier = 3
    stemmer = 4
    save_sa_results(dataset, stemmer, classifier)

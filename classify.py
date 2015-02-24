import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC


def runclassifier(testfile, trainfile):
    trtextfield = input('Training set: Enter name of text field... ')
    trcatfield = input('Training set: Enter name of category field.... ')
    df = pd.read_csv(trainfile)
    traintext = pd.Series(df[trtextfield])
    #testset =  go back and make search speciific, not project...
    train_corpus = []
    #trans_table = ''.join([chr(i) for i in range(128)] + [' '] * 128 )
    for data in traintext:
        #text = text.translate(trans_table)
        train_corpus.append(data)
    i = len(train_corpus)
    z = []
    y = np.zeros(i)
    i = 0
    categories = pd.Series(df[trcatfield])
    for data in categories:
        z.append(data)
    for x in z:
        cat = data.classification.classification_name
        if cat == 'yes':
            y[i] = 0
        if cat == 'no':
            y[i] = 1
        i += 1
    test_corpus = []
    testdf = pd.read_csv(testfile)
    print('testdf', len(testdf))
    testtext = testdf['sm_content']
    i = 0
    for post in testtext:
        text = testtext[i]
        #text = text.translate(trans_table)
        test_corpus.append(text)
        i += 1
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
    X_train = vectorizer.transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)
    clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
    clf.fit(X_train, y)
    pred = clf.predict(X_test)
    print('predlen', len(pred))
    xsmobj = xsmdata.objects.filter(sm_search_id=search_id)
    print('dset len', len(xsmobj))
    i = 0
    # below assumes order of data does not change.
    # this code also assumes that you are doingall the data at once.
    positive_pks = set()
    negative_pks = set()
    for post in xsmobj:
        if pred[i] == 1:
            positive_pks.add(post.pk)
        elif pred[i] == 0:
            negative_pks.add(post.pk)
        i += 1
    positive_qs = xsmdata.objects.filter(pk__in=positive_pks)
    positive_qs.update(sm_sentiment=1)
    negative_qs = xsmdata.objects.filter(pk__in=negative_pks)
    negative_qs.update(sm_sentiment=-1)
    return render(request, 'smclassify/runclassifier.html')


if __name__ == '__main__':
    train = input('Enter name of training set - a csv file...  ')
    test = input('Enter name of test set - a csv file...      ')
    runclassifier(test, train)

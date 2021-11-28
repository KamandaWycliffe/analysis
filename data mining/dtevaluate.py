def evaluate(predictions):
    import pandas as pd
    from pprint import pprint
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    import numpy as np
    preds = pd.read_csv(str(predictions))

    def compute_tp_tn_fn_fp(y_act, y_pred):
        '''
        True positive - actual = 1, predicted = 1
        False positive - actual = 1, predicted = 0
        False negative - actual = 0, predicted = 1
        True negative - actual = 0, predicted = 0
        '''
        tp = len(preds[(y_act == ">50K") & (y_pred == ">50K")])
        tn = len(preds[(y_act == '<=50K') & (y_pred == '<=50K')])
        fn = len(preds[(y_act == '>50K') & (y_pred == '<=50K')])
        fp = len(preds[(y_act == '<=50K') & (y_pred == '>50K')])
        return tp, tn, fp, fn



    tp, tn, fp, fn = compute_tp_tn_fn_fp(preds.actual_income, preds.predicted_income)
    N = tp+ tn+ fp+ fn


    conf_matrix = np.array([[tn,fp],[fn,tp]])
    print('=============================================')
    print("Performance of the Model on the test set")
    print('=============================================')
    print('Confusion Matrix:')
    print('\n')
    pprint(conf_matrix)

    print('\n\nTP for the model:', tp)
    print('TN for the model:', tn)
    print('FP for the model:', fp)
    print('FN for the model:', fn)


    # Classification Error: Overall, how does the classifier predict incorrectly (Misclassification Rate)

    print("\nMisclassification Rate: ",((fp + fn) / N)*100, "%")


    # Classification Accuracy

    print("\nClassification Accuracy:", ((tp +tn) / N)*100, "%")

    # Recall: When the actual value is positive, how often is the prediction correct?
    recall = tp / float(tp + fn)
    print("\nRecall:",tp / float(tp + fn))


    # Precision: When a positive value is predicted, how often is the prediction correct?
    precision = tp / float(tp + fp)

    # F1 Score

    print("\nF1 Score:", (2 * (precision * recall) / (precision + recall)))
    print('=============================================')
if __name__ == '__main__':
    import sys
    predictions = str(sys.argv[1])
    evaluate(predictions)
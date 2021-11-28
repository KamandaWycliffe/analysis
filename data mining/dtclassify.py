def predict(example, model):
            question = list(model.keys())[0]
            feature_name, comparison_operator, value = question.split(" ")

            # ask question
            if comparison_operator == "<=":  # feature is continuous
                if example[feature_name] <= float(value):
                    answer = model[question][0]
                else:
                    answer = model[question][1]

            # feature is categorical
            else:
                if str(example[feature_name]) == value:
                    answer = model[question][0]
                else:
                    answer = model[question][1]

            # base case
            if not isinstance(answer, dict):
                return answer

            # recursive part
            else:
                residual_model = answer
                return predict(example, residual_model)

def classify(model, testfile, predictions):


        import ast
        import pandas as pd
        import numpy as np
        #Import model building function and the gini_index function
        from dtbuild import DecisionTree, DTree, gini_index,check_purity, maximum_information_gain_split, classify_data, determine_type_of_feature, determine_best_split, calculate_overall_entropy,calculate_entropy, split_data, get_potential_splits
        data = pd.read_csv(testfile)
        train = pd.read_csv('train.csv')
        test = pd.read_csv(testfile)
        res = [int(i) for i in model if i.isdigit()]
        try:
            minfreq = res[0]
        except:
            minfreq = None
        min_samples_split = 20
        if minfreq == 1:
            minfreq = minfreq + 1
        else:
            minfreq = minfreq
        min_information_gain  = 1e-5
        model = DTree(train,minfreq = minfreq)

        preds = test.apply(predict, axis=1, args=(model,))
        predictions1 = pd.DataFrame()
        predictions1['actual_income'] = data.income
        predictions1['predicted_income'] = preds
        predictions1.to_csv(predictions, index = False)
if __name__ == '__main__':
    import sys
    model = sys.argv[1]
    testfile = sys.argv[1:][1]
    predictions = sys.argv[1:][2]
    classify(model, testfile, predictions)
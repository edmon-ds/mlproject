model_report = {'RandomForest': 0.85, 'LinearRegression': 0.75, 'SVM': 0.82}
best_model_name = max(model_report , key = model_report.get)
print(best_model_name) 
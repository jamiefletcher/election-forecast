import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def feature_select(X, y, n_trees = 10, random=None):
    print("- Use random forest classifier to select best features ...")
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=random)
    select = SelectFromModel(estimator=rf).fit(X, y)
    X_reduced = select.transform(X)
    X_cols = select.get_feature_names_out()
    return pd.DataFrame(data=X_reduced, columns=X_cols)


def model_select(ids, X, y, models, test_size=0.2, random=None, verbose=False):
    class_labels = pd.unique(y)
    results = []

    print(
        f"- Split dataset into training ({100.0 - test_size*100:.2f}%)"
        f" and test ({test_size*100:.2f}%) ..."
    )
    id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(
        ids, X, y, test_size=test_size, shuffle=True, random_state=random
    )

    print(f"- Evaluate classification performance of {len(models)} models ...")
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pred_train, labels=class_labels)
        f1_train = f1_score(y_train, y_pred_train, average="micro")

        y_pred_test = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_pred_test, labels=class_labels)
        f1_test = f1_score(y_test, y_pred_test, average="micro")

        print(
            f" {model_name.ljust(8, ' ')} \t"
            f" F1 (Train) = {f1_train:.4f} \t F1 (Test) = {f1_test:.4f}"
        )
        results.append({"model": model, "f1_test": f1_test})
        if verbose:
            print("Confusion matrix - Training set")
            print(cm_train, end="\n\n")
            print("Confusion matrix - Test set")
            print(cm_test, end="\n\n")

    return sorted(results, key=lambda d: d["f1_test"], reverse=True)
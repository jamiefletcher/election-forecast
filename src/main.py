from datetime import date

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from process_data import merge_dfs, prepare_census, prepare_elections
from train import feature_select, model_select

# 2021 Census - (Old) 2013 Ridings
census_2013r = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"

# National Election Results - 2015, 2019, 2021
riding_results = {
    2021: "data/Results/44th_table_tableau12.csv",
    2019: "data/Results/43rd_table_tableau12.csv",
    2015: "data/Results/42nd_table_tableau12.csv",
}
national_results = {
    2021: "data/Results/44th_table_tableau09.csv",
    2019: "data/Results/43rd_table_tableau09.csv",
    2015: "data/Results/42nd_table_tableau09.csv",
}

# Model Types to Compare
models = {
    "ridge": RidgeClassifier(),
    "logit": LogisticRegression(max_iter=2000),
    "rf": RandomForestClassifier(ccp_alpha=0.001),
    "svc": SVC(),
    "mlp": MLPClassifier(hidden_layer_sizes=(50, 20)),
}

# Test-Train Split
test_size = 0.2


def data_prep(df_census, df_ridings, df_national):
    dataset = merge_dfs(
        df_census, df_ridings, df_national, target_class="winner", merge_class="id"
    )
    ids = dataset["id"]
    X = dataset.drop(["id", "winner"], axis=1)
    y = dataset["winner"]
    return ids, X, y


def train_model(ids, X, y):
    X_select = feature_select(X, y)
    selection_results = model_select(
        ids, X_select, y, models, test_size=test_size, verbose=True
    )
    return selection_results[0]["model"], X_select.columns


# def polls_predict(model, df_census, df_ridings, df_national, best_features, base_year):
#     this_year = int(date.today().strftime("%Y"))
#     df_ridings_predict = {this_year: df_ridings[base_year]}

#     # TODO Replace this with current poll average
#     df_national_predict = {this_year: df_national[base_year]}

#     dataset = merge_dfs(
#         df_census,
#         df_ridings_predict,
#         df_national_predict,
#         target_class="winner",
#         merge_class="id",
#     )

#     ids = dataset["id"]
#     X = dataset[best_features]
#     y = model.predict(X)
#     return ids, y


def main():
    print("Part I. Load Data")
    df_census = prepare_census(census_2013r)
    df_ridings, df_national = prepare_elections(riding_results, national_results)

    print("Part II. Data Preparation")
    ids, X, y = data_prep(df_census, df_ridings, df_national)

    print("\nPart III. Feature and Model Selection")
    best_model, best_features = train_model(ids, X, y)

    # print("\nPart IV. Get Latest Polling and Predict")
    # ids_predict, y_predict = polls_predict(
    #     best_model, df_census, df_ridings, df_national, best_features, base_year=2021
    # )


if __name__ == "__main__":
    main()

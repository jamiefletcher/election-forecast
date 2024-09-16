from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from process_data import merge_dfs, prepare_census, prepare_elections
from train import feature_select, model_select

# Filepaths
CENSUS_2013R = "data/CanCensus2021_2013Ridings/98-401-X2021010_English_CSV_data.csv"
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

# Constants
TEST_SIZE = 0.2

models = {
    "ridge": RidgeClassifier(),
    "logit": LogisticRegression(max_iter=2000),
    "rf": RandomForestClassifier(ccp_alpha=0.001),
    "svc": SVC(),
    "mlp": MLPClassifier(hidden_layer_sizes=(50, 20)),
}


def main():
    df_census = prepare_census(CENSUS_2013R)
    df_elections = prepare_elections(riding_results, national_results)
    dataset = merge_dfs(
        df_census, df_elections, target_class="winner", merge_class="id"
    )

    ids = dataset["id"]
    X = dataset.drop(["id", "winner"], axis=1)
    y = dataset["winner"]

    X_select = feature_select(X, y)
    model_select(ids, X_select, y, models, test_size=TEST_SIZE)


if __name__ == "__main__":
    main()

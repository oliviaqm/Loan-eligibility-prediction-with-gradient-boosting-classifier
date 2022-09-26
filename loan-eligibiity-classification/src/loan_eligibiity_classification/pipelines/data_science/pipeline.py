from kedro.pipeline import Pipeline, node, pipeline
from .nodes import binarize_target_var, feature_scaling, split_data, train_model, evaluate_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=binarize_target_var,
                inputs=["data"],
                outputs="y",
                name="binarize_target_node",
            ),
            node(
                func=feature_scaling,
                inputs=["preprocessed_data_4"],
                outputs="scaled_features",
                name="feature_scaling_node",
            ),
            node(
                func=split_data,
                inputs=["scaled_features", "y", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:hyperparams"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_models,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:hyperparams"],
                outputs=["models_report", "y_pred", "y_pred_proba"],
                name="evaluate_model_node",
            ),

        ]
    )
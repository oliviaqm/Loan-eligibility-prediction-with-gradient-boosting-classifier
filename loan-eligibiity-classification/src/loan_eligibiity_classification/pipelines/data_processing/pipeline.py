from kedro.pipeline import Pipeline, node, pipeline
from .nodes import replace_outliers,  standardize_class_labels, convert_cat_to_num, replace_missing_values 

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=create_small_dataset,
            #     inputs="data",
            #     outputs="small_data",
            #     name="create_small_data_node",
            # ),
            node(
                func=replace_outliers,
                inputs="data",
                outputs="preprocessed_data_1",
                name="replace_outliers_node",
            ),
            node(
                func=standardize_class_labels,
                inputs="preprocessed_data_1",
                outputs="preprocessed_data_2",
                name="standardize_class_labels_node",
            ),
            node(
                func=convert_cat_to_num,
                inputs="preprocessed_data_2",
                outputs="preprocessed_data_3",
                name="convert_cat_to_num_node",
            ),
            node(
                func=replace_missing_values,
                inputs="preprocessed_data_3",
                outputs="preprocessed_data_4",
                name="replace_missing_values_node",
            ),
        ]
    )

from kedro.pipeline import Pipeline, node

from spaceflights.pipelines.user_app.business_logic import predict_with_mlflow


def create_pipeline(**kwargs):
    pipeline_user_app = Pipeline(
        [
            node(
                func=predict_with_mlflow,
                inputs=dict(model="pipeline_inference_model", data="instances"),
                outputs="predictions",
                tags="user_app",
            )
        ]
    )

    return pipeline_user_app
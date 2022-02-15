job_config = {}


def add_config(
    name,
    train_task_name,
    model_name,
    agg_name,
    train_dataset_name,
    train_collate_fn_name=None,
    test_task_name="TestTask",
    news_dataset_name="NewsDataset",
    user_dataset_name="UserDataset",
):
    job_config[name] = {}
    job_config[name]["train_task_name"] = train_task_name
    job_config[name]["model_name"] = model_name
    job_config[name]["agg_name"] = agg_name
    job_config[name]["train_dataset_name"] = train_dataset_name
    job_config[name]["train_collate_fn_name"] = train_collate_fn_name
    job_config[name]["test_task_name"] = test_task_name
    job_config[name]["news_dataset_name"] = news_dataset_name
    job_config[name]["user_dataset_name"] = user_dataset_name


add_config(
    name="AdverItemNorm",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-MultiKrum",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="MultiKrumAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-NormBound",
    train_task_name="AdverItemNormTrainTask",
    model_name="NRMS",
    agg_name="NormBoundAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

# ========================== LSTUR ===================================
add_config(
    name="AdverItemNorm-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="UserAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-MultiKrum-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="MultiKrumAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)
add_config(
    name="AdverItemNorm-NormBound-LSTUR",
    train_task_name="AdverItemNormTrainTask",
    model_name="LSTUR",
    agg_name="NormBoundAggregator",
    train_dataset_name="TrainMPDataset",
    train_collate_fn_name="train_mp_collate_fn",
)

if __name__ == "__main__":
    print(job_config)

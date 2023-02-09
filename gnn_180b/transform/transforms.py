from tqdm import tqdm


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    if transform_func is None:
        return dataset

    data_list = [
        transform_func(dataset.get(i))
        for i in tqdm(
            range(len(dataset)),
            disable=not show_progress,
            mininterval=10,
            miniters=len(dataset) // 20,
        )
    ]
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

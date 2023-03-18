def is_eval_epoch(epoch: int, max_epochs: int, eval_period: int) -> bool:
    return (
        (epoch + 1) % eval_period == 0
        or epoch == 0
        or (epoch + 1) == max_epochs
    )


def get_each_data_from_batch(data_list: list) -> list:
    out = []
    for batch in data_list:
        out.append(batch.to_data_list())

    return sum(out, [])

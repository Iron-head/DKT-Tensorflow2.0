import pandas as pd
import numpy as np
import tensorflow as tf

MASK_VALUE = -1.0


def load_dataset(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn, encoding="ISO-8859-1")

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")
    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # remove the questions without skill
    df.dropna(subset=['skill_id'], inplace=True)
    # remove the users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()
    # Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
    # Cross skill id with answer to form a synthetic feature
    # TODO: what that means?
    df['skill_with_answer'] = df['skill']*2 + df['correct']
    # Convert to a sequence per user id and shift features 1 time step
    # TODO: why skill_with_answer is from 0 to -1 and the others(skill and correct) is from 1 to last
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
        )
    )
    n_users = len(seq)

    # Get Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_users)

    # Encode categorical features and merge skills with labels to compute target loss
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max()+1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = n_users // batch_size
    return dataset, length, features_depth, skill_depth


# dataset, length, features_depth, skill_depth = load_dataset("skill_builder_data.csv")
# iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()
# print(one_element)


def split_dataset(dataset, total_size, test_fraction=0.2, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)
    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get skills nad labels form y_true
    mask = 1.0 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

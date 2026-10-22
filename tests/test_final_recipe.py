"""Smoke tests for src.data.final_recipe."""

import pandas as pd

from src.data import extended_training, final_recipe


def _half(stem, names, condition):
    return pd.DataFrame(
        {
            "filepath": [f"${{DATA_ROOT}}/clips/{stem}{n}.wav" for n in names],
            "label": "bonafide",
            "speaker": "s",
            "condition": condition,
            "utt_id": names,
            "split": "train",
        }
    )


def test_build_stacks_the_extended_set_with_its_amr_copy(tmp_path):
    _half("", ["a", "b", "c"], "channel_g711_20db").to_csv(
        tmp_path / f"{extended_training.OUT}.csv", index=False
    )
    for stem, names in zip(final_recipe.AMR_PARTS, (["a"], ["b"], ["c"]), strict=True):
        _half("amr/", names, "amr_nb_12k2").to_csv(tmp_path / f"{stem}.csv", index=False)
    out = final_recipe.build(str(tmp_path))
    assert len(out) == 6
    assert out["condition"].value_counts().to_dict() == {"channel_g711_20db": 3, "amr_nb_12k2": 3}

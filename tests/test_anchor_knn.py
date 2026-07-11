import numpy as np
import pytest

from src.anchors.knn import prompt_aware_knn_predict


def test_prompt_knn_excludes_same_id_and_other_prompt() -> None:
    result = prompt_aware_knn_predict(
        query_ids=["self"],
        query_prompt_hashes=["p1"],
        query_embeddings=np.asarray([[1.0, 0.0]]),
        bank_ids=["self", "same_prompt", "other_prompt"],
        bank_prompt_hashes=["p1", "p1", "p2"],
        bank_embeddings=np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]]),
        bank_scores=np.asarray([[5.0, 5.0, 5.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0]]),
        k=3,
    )
    assert result.neighbor_ids == (("same_prompt",),)
    np.testing.assert_allclose(result.predictions, [[2.0, 3.0, 4.0]])
    assert not result.used_global_fallback[0]


def test_unknown_prompt_global_fallback_is_explicit() -> None:
    result = prompt_aware_knn_predict(
        query_ids=["query"],
        query_prompt_hashes=["unknown"],
        query_embeddings=np.asarray([[1.0, 0.0]]),
        bank_ids=["a"],
        bank_prompt_hashes=["known"],
        bank_embeddings=np.asarray([[1.0, 0.0]]),
        bank_scores=np.asarray([[3.0, 3.0, 3.0]]),
        k=1,
        unknown_prompt_policy="global",
    )
    assert result.used_global_fallback[0]
    assert result.neighbor_ids == (("a",),)


def test_unknown_prompt_error_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="no same-prompt anchors"):
        prompt_aware_knn_predict(
            query_ids=["query"],
            query_prompt_hashes=["unknown"],
            query_embeddings=np.asarray([[1.0, 0.0]]),
            bank_ids=["a"],
            bank_prompt_hashes=["known"],
            bank_embeddings=np.asarray([[1.0, 0.0]]),
            bank_scores=np.asarray([[3.0, 3.0, 3.0]]),
            k=1,
            unknown_prompt_policy="error",
        )

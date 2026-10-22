import pytest

from app.retrieval.embeddings import model_prefixes, vector_literal
from app.retrieval.retriever import rrf_scores


def test_rrf_scores_combine_both_branches():
    scores = rrf_scores([1, 2], [2, 3], k=60)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert scores[1] < scores[2]


def test_e5_prefixes():
    assert model_prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")


def test_potion_has_no_prefixes():
    assert model_prefixes("minishlab/potion-multilingual-128M") == ("", "")


def test_vector_literal_is_pgvector_compatible():
    assert vector_literal([0.1, -0.25, 1.0]) == "[0.1000000,-0.2500000,1.0000000]"

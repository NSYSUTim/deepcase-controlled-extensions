from __future__ import annotations

import numpy as np

from context_selection_experiment.run_gate_b import (
    IncidentAlignedContextBuilder,
    _direct_attention_batch,
)
from research_experiments.src.model import build_attended_cache, set_reproducible_seed


def test_direct_zero_iteration_cache_matches_original_adapter():
    set_reproducible_seed(17)
    model = IncidentAlignedContextBuilder(input_size=6, hidden_size=4, max_length=3)
    context = np.asarray([[5, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=np.int16)
    target = np.asarray([2, 3, 4], dtype=np.int16)
    original = build_attended_cache(
        model,
        context,
        target,
        None,
        features=6,
        query_iterations=0,
        preserve_rows=True,
        batch_size=2,
    )
    confidence, vectors = _direct_attention_batch(model, context, target, features=6)
    np.testing.assert_allclose(confidence, original.confidence, atol=1e-7, rtol=0)
    np.testing.assert_allclose(vectors.toarray(), original.vectors.toarray(), atol=1e-7, rtol=0)

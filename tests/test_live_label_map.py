# tests/test_live_label_map.py
#
# Unit tests for default ANUBIS class-name map resolution in live inference.
# -------------------------------------------------------------------------
# Run with: pytest tests/test_live_label_map.py -v

from __future__ import annotations

from module1.pretrain.live_inference import _resolve_label_map_path
from module1.pretrain.live_utils import load_class_names


class TestLiveLabelMap:
    def test_resolve_builtin_map_for_anubis_102(self) -> None:
        path = _resolve_label_map_path(None, num_classes=102)
        assert path is not None
        names = load_class_names(102, path)
        assert len(names) == 102
        assert names[0] == "hit with knees"
        assert names[101] == "cheers and drink"

    def test_no_default_for_non_102(self) -> None:
        path = _resolve_label_map_path(None, num_classes=10)
        assert path is None

    def test_user_label_map_takes_priority(self) -> None:
        custom = "custom/path/labels.yaml"
        path = _resolve_label_map_path(custom, num_classes=102)
        assert path == custom

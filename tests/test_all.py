# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


def test_import():
    import blur_detection
    from blur_detection import fix_image_size
    from blur_detection import estimate_blur
    from blur_detection import pretty_blur_map
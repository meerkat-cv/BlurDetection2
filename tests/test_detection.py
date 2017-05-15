# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import numpy.linalg
import pytest

import blur_detection


def test_fix_image_size():
    image = numpy.random.randint(0, 255, (1920, 1080, 3), dtype=numpy.uint8)
    image_out = blur_detection.fix_image_size(image)
    assert type(image_out) == numpy.ndarray
    assert image_out.ndim == image.ndim
    assert (1852, 1042, 3) == image_out.shape

    image = numpy.random.randint(0, 255, (960, 940, 2), dtype=numpy.uint8)
    image_out = blur_detection.fix_image_size(image)
    assert type(image_out) == numpy.ndarray
    assert image_out.ndim == image.ndim
    assert (960, 940, 2) == image_out.shape
    assert abs(numpy.linalg.norm(image_out - image)) < 1E-4

    image = numpy.random.randint(0, 255, (960, 940, 2), dtype=numpy.uint8)
    image_out = blur_detection.fix_image_size(image, 9E5)
    assert type(image_out) == numpy.ndarray
    assert image_out.ndim == image.ndim
    assert (957, 938, 2) == image_out.shape


def test_estimate_blur():
    image = numpy.zeros((1920, 1080, 3), dtype=numpy.uint8)
    blur_map, score, is_blurry = blur_detection.estimate_blur(image)

    assert type(blur_map) == numpy.ndarray
    assert type(score) == numpy.float64
    assert type(is_blurry) == bool

    assert blur_map.ndim == 2
    assert blur_map.shape == image.shape[:2]

    assert abs(score) < 1E-3
    assert is_blurry == True


def test_pretty_blur_map():
    image = numpy.random.rand(1920, 1080, 3)
    display_image = blur_detection.pretty_blur_map(image)

    assert type(display_image) == numpy.ndarray
    assert image.shape[0] == display_image.shape[0]
    assert image.shape[1] == display_image.shape[1]
    assert image.shape[2] == display_image.shape[2]

    image = numpy.random.rand(1920, 1080)
    display_image = blur_detection.pretty_blur_map(image)

    assert type(display_image) == numpy.ndarray
    assert image.shape[0] == display_image.shape[0]
    assert image.shape[1] == display_image.shape[1]

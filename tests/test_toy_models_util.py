from antorcha.toy_models.util import estimate_conv2d_size


def test_estimate_conv2d_size():
    assert estimate_conv2d_size(8, []) == 8
    assert estimate_conv2d_size(8, [2]) == 4
    assert estimate_conv2d_size(8, [2, 2]) == 2

    assert estimate_conv2d_size(60, [5, 4, 3]) == 1
    assert estimate_conv2d_size(60, [5, 3, 2]) == 2
    assert estimate_conv2d_size(28, [2, 2, 2]) == 4


def test_estimate_conv2d_size_with_upsampling():
    assert estimate_conv2d_size(60, [5, 3, 2], [2, 2, 2]) == 16
    assert estimate_conv2d_size(60, [5, 3, 2], [2, 3, 5]) == 60

    # unequal length of strides and up_sample
    assert estimate_conv2d_size(60, [5, 3, 2], [2]) == 24
    assert estimate_conv2d_size(60, [3], [4, 3, 2]) == 80

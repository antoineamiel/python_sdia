import numpy as np
from lab2.ipynb import gradient2D

# Test case 1: constant matrix (should give zero gradients)
def test_gradient2D_constant_matrix():
    constant_matrix = np.ones((3, 4))
    Xh, Xv = gradient2D(constant_matrix)

    # Check output shapes
    assert Xh.shape == constant_matrix.shape, "Horizontal gradient has wrong shape"
    assert Xv.shape == constant_matrix.shape, "Vertical gradient has wrong shape"

    # Check that gradients are zero for constant matrix
    assert np.allclose(Xh, 0), "Horizontal gradient should be zero for constant matrix"
    assert np.allclose(Xv, 0), "Vertical gradient should be zero for constant matrix"

# Test case 2: non-square matrix
def test_gradient2D_simple_matrix():
    test_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    Xh, Xv = gradient2D(test_matrix)

    # Check output shapes for non-square matrix
    assert Xh.shape == test_matrix.shape, "Horizontal gradient has wrong shape"
    assert Xv.shape == test_matrix.shape, "Vertical gradient has wrong shape"

# Test case 3: error for 3D input
def test_gradient2D_3D_input():
    try:
        gradient2D(np.ones((2, 2, 2)))
        assert False, "Should raise error for 3D input"
    except AssertionError:
        pass

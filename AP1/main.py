import numpy as np
import matplotlib.pyplot as plt


# part 1
# defining norm1
def norm_1(x):
    total = 0
    for x1 in x:
        total += abs(x1)
    return total


# defining infinity norm
def norm_inf(x):
    max_value = 0
    for x1 in x:
        if abs(x1) > max_value:
            max_value = abs(x1)
    return max_value


# induced norm for vector norm 1
def matrix_induced_norm_1(A):
    rows, cols = A.shape
    max_col_sum = 0
    for j in range(cols):
        col_sum = 0
        for i in range(rows):
            col_sum += abs(A[i, j])
        if col_sum > max_col_sum:
            max_col_sum = col_sum
    return max_col_sum


# induced norm for vector norm infinity
def matrix_induced_norm_inf(A):
    rows, cols = A.shape
    max_row_sum = 0
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += abs(A[i, j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum
    return max_row_sum


# part 2
np.random.seed(42)  # to make random vectors closer
v1 = np.random.randn(4)
v2 = np.random.randn(4)

print(v1)
print(v2)

A1 = v1.reshape(2, 2)
A2 = v2.reshape(2, 2)

# part 3
vector_difference = v1 - v2
matrix_difference = A1 - A2

# calculating distance using different norms
vector_difference_norm1 = norm_1(vector_difference)
vector_difference_norminf = norm_inf(vector_difference)
matrix_difference_norm1 = matrix_induced_norm_1(matrix_difference)
matrix_difference_norminf = matrix_induced_norm_inf(matrix_difference)


print(f"Vector difference norm 1 : {vector_difference_norm1}")
print(f"vector difference norm infinity : {vector_difference_norminf}")

print(f"matrix difference norm 1 : {matrix_difference_norm1}")
print(f"matrix difference norm infinity : {matrix_difference_norminf}")


# part 4
def plot_matrix_unit_ball(A_ref, induced_norm_func, title):
    a_range = np.linspace(
        A_ref[0, 0] - 1.5, A_ref[0, 0] + 1.5, 150
    )  # range of values for A[0,0]
    b_range = np.linspace(
        A_ref[0, 1] - 1.5, A_ref[0, 1] + 1.5, 150
    )  # range of values for A[0,1]
    A, B = np.meshgrid(a_range, b_range)  # create 2D grids of all (a,b) combinations
    mask = np.zeros_like(
        A, dtype=bool
    )  # initialize boolean grid to mark points inside unit ball
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            M = np.array(
                [[A[i, j], B[i, j]], [A_ref[1, 0], A_ref[1, 1]]]
            )  # fix last two elements for 2d visualization
            diff = M - A_ref  # calculate difference matrix
            if (
                induced_norm_func(diff) <= 1
            ):  # check if induced norm of difference is less or equal then 1
                mask[i, j] = True  # mark this point as inside the unit ball

    plt.contour(A, B, mask, levels=[0.5], colors=["blue"], alpha=0.5)
    plt.plot(A_ref[0, 0], A_ref[0, 1], "o", label=title)


def plot_vector_unit_ball(center, norm_func, title):
    grid_size = 200
    x1 = np.linspace(center[0] - 1.5, center[0] + 1.5, grid_size)  # range values for x1
    x2 = np.linspace(center[1] - 1.5, center[1] + 1.5, grid_size)  # range values for x2
    X1, X2 = np.meshgrid(x1, x2)  # 2d grid for all combinations

    x3, x4 = center[2], center[3]  # Fix the other two components
    mask = np.zeros_like(X1, dtype=bool)  # boolean mask

    for i in range(grid_size):
        for j in range(grid_size):
            x = np.array([X1[i, j], X2[i, j], x3, x4])  # fix last 2 points
            diff = x - center  # calculate difference vector
            if (
                norm_func(diff) <= 1
            ):  # if norm is less or equal then one make it true so we can plot it later
                mask[i, j] = True

    plt.contour(X1, X2, mask, levels=[0.5], colors=["black"], alpha=0.5)
    plt.plot(center[0], center[1], "o", label=title)


# plot_vector_unit_ball(v1, norm_1, "Unit Ball around v1 (1-norm)")
# plot_vector_unit_ball(v2, norm_inf, "Unit Ba")


# plot_matrix_unit_ball(A1, matrix_induced_norm_1, "Matrix A1 Unit Ball (induced 1-norm)")
# plot_matrix_unit_ball(
#     A2, matrix_induced_norm_inf, "Matrix A2 Unit Ball (induced ∞-norm)"
# )


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

# vectors
plt.sca(ax1)
plot_vector_unit_ball(v1, norm_1, "v1 (1-norm)")
plot_vector_unit_ball(v2, norm_inf, "v2 (∞-norm)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.legend()
plt.title("Vector Unit Balls")

# matrices
plt.sca(ax2)
plot_matrix_unit_ball(A1, matrix_induced_norm_1, "A1 (1-norm)")
plot_matrix_unit_ball(A2, matrix_induced_norm_inf, "A2 (∞-norm)")
plt.xlabel("a11")
plt.ylabel("a12")
plt.axis("equal")
plt.legend()
plt.title("Matrix Unit Balls")

plt.tight_layout()
plt.show()

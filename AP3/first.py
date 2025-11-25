import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd


# one variable function
def f(x):
    return x**3 * np.cos(x)


def f_exact_derivative(x):
    return 3 * x**2 * np.cos(x) - x**3 * np.sin(x)


# two variable function
def g(x, y):
    return np.exp(-(x**2 + y**2)) * np.sin(2 * x) * np.cos(2 * y)


# exact gradient of g
def g_exact_gradient(x, y):
    dg_dx = np.exp(-(x**2 + y**2)) * (  # product rule
        2 * np.cos(2 * x) * np.cos(2 * y) - 2 * x * np.sin(2 * x) * np.cos(2 * y)
    )
    dg_dy = np.exp(-(x**2 + y**2)) * (
        -2 * np.sin(2 * x) * np.sin(2 * y) - 2 * y * np.sin(2 * x) * np.cos(2 * y)
    )
    return dg_dx, dg_dy


h_value = 1e-2  # small


# central finite difference for derivative
def finite_difference_derivative(f, x, h=h_value):
    return (f(x + h) - f(x - h)) / (2 * h)


# central finite difference for gradient in two variables
def finite_difference_2d(g, x0, y0, h=h_value):
    gx = (g(x0 + h, y0) - g(x0 - h, y0)) / (2 * h)
    gy = (g(x0, y0 + h) - g(x0, y0 - h)) / (2 * h)
    return gx, gy


# for 1d

# direction of tangent line [1, f'(x0)]
# normal vector to tangent line [-f'(x0), 1]

x1 = 0.5
y1 = f(x1)
slope_exact_1d = f_exact_derivative(x1)


tangent_line_1d = np.array([1, slope_exact_1d])
normal_vector_1d = np.array([-slope_exact_1d, 1])
normal_vector_1d = normal_vector_1d / np.linalg.norm(normal_vector_1d)  # normalize
print("1D Normal Vector:", normal_vector_1d)

slope_finite_diff_1d = finite_difference_derivative(f, x1)
normal_vector_using_finite_diff_1d = np.array([-slope_finite_diff_1d, 1])
normal_vector_using_finite_diff_1d = (
    normal_vector_using_finite_diff_1d
    / np.linalg.norm(normal_vector_using_finite_diff_1d)
)  # normalize
print("1D Normal Vector (Finite Difference):", normal_vector_using_finite_diff_1d)


# for 2d


# direction of tangent plane  z-z0=gx*(x-x0)+gy*(y-y0) point (x0,y0,z0) z=g(x0,y0)
# tangent in x direction [1,0,gx] tangent in y direction [0,1,gy]
# normal vector to tangent plane [gx,gy,-1]

x2 = 0.5
y2 = -0.3
z2 = g(x2, y2)

gx, gy = g_exact_gradient(x2, y2)

tangent_plane_2d_x = np.array([1, 0, gx])
tangent_plane_2d_y = np.array([0, 1, gy])
normal_vector_2d = np.array([gx, gy, -1])
normal_vector_2d = normal_vector_2d / np.linalg.norm(normal_vector_2d)  # normalize
print("2D Normal Vector:", normal_vector_2d)

gx_finite_diff, gy_finite_diff = finite_difference_2d(g, x2, y2)
normal_vector_using_finite_diff_2d = np.array([gx_finite_diff, gy_finite_diff, -1])
normal_vector_using_finite_diff_2d = (
    normal_vector_using_finite_diff_2d
    / np.linalg.norm(normal_vector_using_finite_diff_2d)
)
print("2D Normal Vector (Finite Difference):", normal_vector_using_finite_diff_2d)


# now calculating accuracy

h_values = np.logspace(-8, -1, 10)  # from 1e-8 to 1e-1

error_1d = []
error_2d = []

for h in h_values:
    # 1d case
    slope_finite_diff_1d = finite_difference_derivative(f, x2, h)
    normal_vector_using_finite_diff_1d = np.array([-slope_finite_diff_1d, 1])
    normal_vector_using_finite_diff_1d = (
        normal_vector_using_finite_diff_1d
        / np.linalg.norm(normal_vector_using_finite_diff_1d)
    )
    err1 = np.linalg.norm(
        normal_vector_using_finite_diff_1d - normal_vector_1d
    )  # error
    error_1d.append(err1)

    # 2d case
    gx_finite_diff, gy_finite_diff = finite_difference_2d(g, x2, y2, h)
    normal_vector_using_finite_diff_2d = np.array([gx_finite_diff, gy_finite_diff, -1])
    normal_vector_using_finite_diff_2d = (
        normal_vector_using_finite_diff_2d
        / np.linalg.norm(normal_vector_using_finite_diff_2d)
    )
    err2 = np.linalg.norm(
        normal_vector_using_finite_diff_2d - normal_vector_2d
    )  # error
    error_2d.append(err2)


table = pd.DataFrame(
    {"h": h_values, "1D Normal Error": error_1d, "2D Normal Error": error_2d}
)
print(table)  # display table

# same for tangent vectors
error_1d_tangent = []
error_2d_tangent = []

for h in h_values:
    # 1d case
    slope_finite_diff_1d = finite_difference_derivative(f, x1, h)
    tangent_vector_1d = np.array([1, slope_finite_diff_1d])
    tangent_vector_1d = tangent_line_1d / np.linalg.norm(tangent_vector_1d)

    exact_tangent_1d = tangent_line_1d / np.linalg.norm(tangent_line_1d)
    err1 = np.linalg.norm(tangent_vector_1d - exact_tangent_1d)
    error_1d_tangent.append(err1)

    # 2d case
    gx_finite_diff, gy_finite_diff = finite_difference_2d(g, x2, y2, h)

    tangent_vector_2d_x = np.array([1, 0, gx_finite_diff])
    tangent_vector_2d_y = np.array([0, 1, gy_finite_diff])

    tangent_vector_2d_x = tangent_plane_2d_x / np.linalg.norm(tangent_vector_2d_x)
    tangent_vector_2d_y = tangent_plane_2d_y / np.linalg.norm(tangent_vector_2d_y)

    exact_tangent_2d_x = tangent_plane_2d_x / np.linalg.norm(tangent_plane_2d_x)
    exact_tangent_2d_y = tangent_plane_2d_y / np.linalg.norm(tangent_plane_2d_y)
    err2 = (
        np.linalg.norm(tangent_vector_2d_x - exact_tangent_2d_x)
        + np.linalg.norm(tangent_vector_2d_y - exact_tangent_2d_y)
    ) / 2  # average error
    error_2d_tangent.append(err2)


table_tangent = pd.DataFrame(
    {
        "h": h_values,
        "1D Tangent Error": error_1d_tangent,
        "2D Tangent Error": error_2d_tangent,
    }
)
print(table_tangent)


plt.figure(figsize=(8, 5))
plt.loglog(h_values, error_1d, "o-", label="1D Normal Error")
plt.loglog(h_values, error_2d, "s-", label="2D Normal Error")
plt.xlabel("Step size h")
plt.ylabel("Error (||exact - finite-diff||)")
plt.title("Accuracy of Finite Difference Normal Vectors")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.loglog(h_values, error_1d_tangent, "o-", label="1D Tangent Error")
plt.loglog(h_values, error_2d_tangent, "s-", label="2D Tangent Error")
plt.xlabel("Step size h")
plt.ylabel("Error (||exact - finite-diff||)")
plt.title("Accuracy of Finite Difference Tangent Vectors")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# 1d tangent line plot
x_vals = np.linspace(x1 - 1, x1 + 1, 100)
y_vals_1d_exact = slope_exact_1d * (x_vals - x1) + y1
y_vals_1d_finite_diff = slope_finite_diff_1d * (x_vals - x1) + y1

plt.figure(figsize=(7, 5))
plt.plot(x_vals, f(x_vals), label="f(x)", color="blue")
plt.plot(x_vals, y_vals_1d_exact, "--", label="Tangent Line (Exact)", color="orange")
plt.plot(
    x_vals,
    y_vals_1d_finite_diff,
    "-.",
    label="Tangent Line (Finite Diff)",
    color="green",
)
plt.scatter(x1, f(x1), color="red", zorder=5, label="Point on curve")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("1D Function and Tangent Lines")
plt.grid(True)
plt.legend()
plt.show()


# 2d tangent plane plot

x_range = np.linspace(x2 - 0.2, x2 + 0.2, 10)
y_range = np.linspace(y2 - 0.2, y2 + 0.2, 10)
X, Y = np.meshgrid(x_range, y_range)

# z = z0 + gx*(x-x0) + gy*(y-y0)

# actual 2D function
z_function = g(X, Y)

# tangent plane at (x2, y2)
z_tangent_exact = z2 + gx * (X - x2) + gy * (Y - y2)
z_function_finite_diff = z2 + gx_finite_diff * (X - x2) + gy_finite_diff * (Y - y2)

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, z_function, alpha=0.6, cmap="viridis")
ax.plot_surface(X, Y, z_tangent_exact, alpha=0.5, color="orange")
ax.plot_surface(X, Y, z_function_finite_diff, alpha=0.4, color="magenta")

ax.scatter(x2, y2, z2, color="red", s=50)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("2D Function and Tangent Plane at (x2, y2)")
plt.legend(["Function Surface", "Tangent Plane (Exact)", "Tangent Plane (Finite Diff)"])
plt.show()

# plotting normal vectors and funciton too
plt.figure(figsize=(7, 7))
origin_1d = np.array([[x1, y1], [0, 0]])  # origin point
plt.quiver(
    origin_1d[0, 0],
    origin_1d[0, 1],
    normal_vector_1d[0],
    normal_vector_1d[1],
    color="blue",
    scale=3,
    label="1D Normal Vector (Exact)",
)

plt.quiver(
    origin_1d[0, 0],
    origin_1d[0, 1],
    normal_vector_using_finite_diff_1d[0],
    normal_vector_using_finite_diff_1d[1],
    color="orange",
    scale=3,
    label="1D Normal Vector (Finite Diff)",
)
x_plot = np.linspace(x1 - 1, x1 + 1, 100)
plt.plot(x_plot, f(x_plot), label="f(x)", color="green")
plt.scatter(x1, y1, color="red", zorder=5, label="Point on curve")
plt.xlim(x1 - 1, x1 + 1)
plt.ylim(y1 - 1, y1 + 1)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("1D Normal Vectors at Point on Curve")
plt.grid()
plt.legend()
plt.show()

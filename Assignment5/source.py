import numpy as np
import matplotlib.pyplot as plt

def intersection(l1, l2):
    """
    Compute the intersection of two lines (in homogeneous coordinates)
    using the cross product.
    """
    p = np.cross(l1, l2)
    # Normalize if possible (if the third coordinate is nonzero)
    if np.abs(p[2]) > 1e-10:
        return p / p[2]
    return p

def normalize_point(p, tol=1e-10):
    """
    Normalize a homogeneous point: if the last coordinate is not zero,
    divide by it. Otherwise, try to normalize by the first nonzero coordinate.
    """
    if np.abs(p[2]) > tol:
        return p / p[2]
    elif np.abs(p[0]) > tol:
        return p / p[0]
    elif np.abs(p[1]) > tol:
        return p / p[1]
    return p

def are_concurrent(l1, l2, l3, tol=1e-6):
    """
    Check if three lines are concurrent.
    They are concurrent if the intersection of l1 and l2 equals
    the intersection of l1 and l3 (up to scaling).
    Returns a tuple (concurrent_bool, intersection_point).
    """
    p12 = intersection(l1, l2)
    p13 = intersection(l1, l3)
    # Normalize both intersection points for comparison
    p12n = normalize_point(p12, tol)
    p13n = normalize_point(p13, tol)
    return np.allclose(p12n, p13n, atol=tol), p12n

def plot_lines_and_intersections(lines, concurrent, inter_points, title=""):
    """
    Plot the three lines given by their homogeneous coordinates and
    plot the pairwise intersection points.
    """
    x_vals = np.linspace(-10, 10, 400)
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b']
    
    # Plot each line
    for i, line in enumerate(lines):
        a, b, c = line
        # if b is not zero, solve for y = (-a*x - c) / b
        if np.abs(b) > 1e-6:
            y_vals = (-a * x_vals - c) / b
            plt.plot(x_vals, y_vals, colors[i], label=f'Line {i+1}: {line}')
        else:
            # vertical line: x = -c/a
            x_line = -c / a
            plt.axvline(x=x_line, color=colors[i], label=f'Line {i+1}: {line}')
    
    # Plot all pairwise intersection points
    for p in inter_points:
        p_norm = normalize_point(p)
        plt.plot(p_norm[0], p_norm[1], 'ko', markersize=8)
    
    # Set title and legend
    if concurrent:
        plt.title(title + " - Lines are concurrent")
    else:
        plt.title(title + " - Lines are NOT concurrent")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    # plt.show()
    
    # save the plot
    plt.savefig("Assets/" + title.replace(" ", "_") + ".png")

def process_case(lines, case_title=""):
    """
    Process one test case: check for concurrency, print the result,
    and generate a plot.
    """
    concurrent_flag, inter_point = are_concurrent(lines[0], lines[1], lines[2])
    # Compute all pairwise intersections
    p12 = intersection(lines[0], lines[1])
    p13 = intersection(lines[0], lines[2])
    p23 = intersection(lines[1], lines[2])
    
    print(case_title)
    print("Concurrent:", concurrent_flag)
    if concurrent_flag:
        print("Intersection point (normalized if possible):", normalize_point(inter_point))
    else:
        print("Pairwise intersections (normalized):")
        for i, p in enumerate([p12, p13, p23], start=1):
            print(f"Intersection {i}: {normalize_point(p)}")
    plot_lines_and_intersections(lines, concurrent_flag, [p12, p13, p23], title=case_title)

# --- Test Cases ---

# Case 1:
# l = (1, 1, -2)T, m = (2, -1, 1)T, n = (-1, 2, 0)T
l1 = np.array([1, 1, -2])
m1 = np.array([2, -1, 1])
n1 = np.array([-1, 2, 0])
lines1 = [l1, m1, n1]
process_case(lines1, "Case 1")

# Case 2:
# l = (2, -1, 1)T, m = (-3, -5, 5)T, n = (-5, -2, 2)T
l2 = np.array([2, -1, 1])
m2 = np.array([-3, -5, 5])
n2 = np.array([-5, -2, 2])
lines2 = [l2, m2, n2]
process_case(lines2, "Case 2")

# Case 3:
# l = (1, 2, 3)T, m = (-2, -4, 0)T, n = (0, 0, 2)T
l3 = np.array([1, 2, 3])
m3 = np.array([-2, -4, 0])
n3 = np.array([0, 0, 2])
lines3 = [l3, m3, n3]
process_case(lines3, "Case 3")

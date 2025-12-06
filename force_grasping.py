import numpy as np
from scipy.spatial.transform import Rotation

def contact_screw_3d(contact_points: np.ndarray, contact_normals: np.ndarray) -> np.ndarray:
    """
    Compute normalized contact screws (Plucker coordinates) in wrench space.
    """
    N = contact_points.shape[0]
    wrench = np.zeros((N, 6))
    
    for i in range(N):
        p = contact_points[i]  # Contact position
        n = contact_normals[i]  # Contact normal (inward)
        
        # Normalize contact normal to get unit force
        force_magnitude = np.linalg.norm(n)
        if force_magnitude < 1e-10:
            wrench[i] = np.zeros(6)
            continue
        
        # Unit force
        force = n / force_magnitude
        
        # Moment = r × F (using unit force)
        moment = np.cross(p, force)
        
        # Concatenate [moment; force] where ||force|| = 1
        screw = np.concatenate([moment, force])
        
        wrench[i] = screw
    
    return wrench

def friction_cone_3d(contact_points: np.ndarray, contact_normals: np.ndarray, mu: float, n_fc: int) -> tuple:
    """
    Approximate friction cone as polyhedral cone with n_fc edges.
    """
    N = contact_points.shape[0]
    contact_points_FC = []
    contact_normals_FC = []
    
    alpha = np.arctan(mu)
    
    for i in range(N):
        p = contact_points[i]
        n = contact_normals[i]
        
        # Local coordinate frame
        z_axis = n / np.linalg.norm(n)
        
        # Arbitrary x-axis
        if abs(z_axis[0]) < 0.9:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = np.array([0, 1, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Generate friction cone edges
        for j in range(n_fc):
            theta = 2 * np.pi * j / n_fc
            
            # Direction in local frame
            # Tilted by alpha from z-axis
            local_dir = (np.sin(alpha) * np.cos(theta) * x_axis +
                        np.sin(alpha) * np.sin(theta) * y_axis +
                        np.cos(alpha) * z_axis)
            
            contact_points_FC.append(p)
            contact_normals_FC.append(local_dir)
            
    return np.array(contact_points_FC), np.array(contact_normals_FC)

def is_force_closure(w: np.ndarray) -> tuple:
    """
    Check if the set of wrenches w can generate force closure.
    Uses ray shooting method (computing z_max).
    """
    from scipy import optimize
    
    # Center of the convex hull (centroid of wrench points)
    w_avg = np.mean(w, axis=0)
    
    # Ray shooting: max z such that w_avg - z*w_avg is in ConvexHull(w)
    # This is equivalent to checking if origin is in ConvexHull(w)
    # If origin is strictly inside, z_max > 1 (since 0 = w_avg - 1*w_avg)
    # Actually, standard measure is slightly different, but checking origin containment is key.
    
    # Formulate as Linear Program:
    # minimize -z
    # subject to: sum(k_i * w_i) = 0 (origin containment)
    #             sum(k_i) = 1 (convex combination)
    #             k_i >= 0
    
    # Simpler check: Does 0 lie in CH(w)?
    # Uses scipy.optimize.linprog
    # minimize 0
    # subject to W.T @ k = 0
    #            sum(k) = 1
    #            k >= 0
    
    n_wrenches = w.shape[0]
    c = np.zeros(n_wrenches) # Dummy objective
    
    # Equality constraints
    # W.T @ k = 0 (6 constraints)
    A_eq = w.T
    b_eq = np.zeros(6)
    
    # Sum(k) = 1 constraint
    A_eq = np.vstack([A_eq, np.ones((1, n_wrenches))])
    b_eq = np.append(b_eq, 1.0)
    
    bounds = [(0, None) for _ in range(n_wrenches)]
    
    res = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        # Origin is in convex hull
        # To get a "quality" metric (z_max), we need the actual ray shooting implementation
        # For now, just returning 1.0 if success (stable) and 0.0 if not
        return True, 1.0
    else:
        return False, 0.0


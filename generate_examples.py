#!/usr/bin/env python3
"""
3D Scene Generator for Photogrammetry Studio
Generates 3 distinct example datasets (Temple, Terrain, Mechanical)
"""

import cv2
import numpy as np
import os
import math

# ═══════════════════════════════════════════════════════════════════════
#  RENDERING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def render_scene(points3d, colors3d, output_dir, num_images=16, cam_distance=10, img_size=800):
    """
    Renders a 3D point cloud scene from multiple camera angles using perspective projection.
    Uses painter's algorithm (depth sorting) to handle occlusion so objects look solid.
    """
    os.makedirs(output_dir, exist_ok=True)
    focal = 600
    cx, cy = img_size / 2, img_size / 2
    
    for i in range(num_images):
        img = np.full((img_size, img_size, 3), 230, dtype=np.uint8)
        
        # Camera orbits around the center
        ang = (i / num_images) * 2 * np.pi
        cam_pos = np.array([
            cam_distance * np.sin(ang),
            cam_distance * 0.4,                # Slight top-down angle
            cam_distance * np.cos(ang)
        ])
        
        # Camera coordinate system
        fwd = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        
        # Project all points to 2D
        d = points3d - cam_pos
        z_cam = d @ fwd
        
        # Filter points behind camera
        valid_mask = z_cam > 0.5
        d_valid = d[valid_mask]
        z_valid = z_cam[valid_mask]
        c_valid = colors3d[valid_mask]
        
        x_cam = d_valid @ right
        y_cam = d_valid @ up
        
        px = (focal * x_cam / z_valid + cx).astype(int)
        py = (-focal * y_cam / z_valid + cy).astype(int)
        
        # Painter's algorithm: sort by depth (farthest first)
        sort_idx = np.argsort(-z_valid)
        px, py, z_valid, c_valid = px[sort_idx], py[sort_idx], z_valid[sort_idx], c_valid[sort_idx]
        
        # Draw points (size based on distance for perspective effect)
        for k in range(len(px)):
            x, y = px[k], py[k]
            if 0 <= x < img_size and 0 <= y < img_size:
                radius = max(2, int(40 / z_valid[k]))
                color = (int(c_valid[k][0]), int(c_valid[k][1]), int(c_valid[k][2]))
                cv2.circle(img, (x, y), radius, color, -1)
        
        # Add subtle noise to simulate real camera sensor
        noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(os.path.join(output_dir, f"img_{i+1:02d}.png"), img)
    
    print(f"✅ Generated {num_images} images in '{output_dir}'")


# ═══════════════════════════════════════════════════════════════════════
#  SCENE GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def generate_temple(output_dir):
    """Example 1: A Greco-Roman temple with columns, base, and roof"""
    pts, cols = [], []
    
    # Ground plane
    for x in np.linspace(-5, 5, 200):
        for z in np.linspace(-5, 5, 200):
            pts.append([x, 0, z])
            cols.append([180 + np.random.randint(20), 170 + np.random.randint(20), 140 + np.random.randint(20)])
            
    # Base Platform
    for x in np.linspace(-3, 3, 150):
        for z in np.linspace(-3, 3, 150):
            for y in np.linspace(0, 0.5, 5):
                if abs(x) > 2.8 or abs(z) > 2.8 or abs(y) < 0.1 or abs(y-0.5) < 0.1:
                    pts.append([x, y, z])
                    cols.append([160, 150, 130])
                    
    # Columns (Cylinders)
    column_positions = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]
    for cx, cz in column_positions:
        for angle in np.linspace(0, 2*np.pi, 30):
            for y in np.linspace(0.5, 3.5, 40):
                r = 0.25
                x_ = cx + r * np.cos(angle)
                z_ = cz + r * np.sin(angle)
                pts.append([x_, y, z_])
                # Marble white with subtle shading
                shade = 200 + int(30 * np.cos(angle))
                cols.append([shade, shade, shade+10])
                
    # Roof
    for x in np.linspace(-3.5, 3.5, 150):
        for z in np.linspace(-3.5, 3.5, 150):
            # Flat roof part
            pts.append([x, 3.5, z])
            cols.append([140, 130, 120])
            # Triangular pediment front/back
            if abs(z - 3.5) < 0.2 or abs(z + 3.5) < 0.2:
                height = max(0, 1.5 - abs(x) * 0.5)
                for y in np.linspace(3.5, 3.5 + height, 10):
                    pts.append([x, y, z])
                    cols.append([140, 130, 120])

    render_scene(np.array(pts, dtype=np.float64), 
                 np.array(cols, dtype=np.float64), 
                 output_dir, cam_distance=12)


def generate_terrain(output_dir):
    """Example 2: A wavy terrain with rocks/boulders"""
    pts, cols = [], []
    
    # Wavy ground
    for x in np.linspace(-6, 6, 300):
        for z in np.linspace(-6, 6, 300):
            y = np.sin(x) * np.cos(z) * 0.8 + np.sin(x*0.5) * 0.5
            pts.append([x, y, z])
            # Grass green with dirt variation
            g_val = 120 + int(40 * (y + 1))
            cols.append([50 + np.random.randint(30), g_val, 30 + np.random.randint(20)])
            
    # Rocks (Spheres partially buried in the ground)
    rock_centers = [(2, 0.5, 3), (-3, 0.2, -2), (0, 0.8, -4), (4, -0.2, -1)]
    for rcx, rcy, rcz in rock_centers:
        for angle_h in np.linspace(0, 2*np.pi, 40):
            for angle_v in np.linspace(-np.pi/2, np.pi/2, 20):
                r = np.random.uniform(0.6, 1.2)
                x_ = rcx + r * np.cos(angle_v) * np.cos(angle_h)
                y_ = rcy + r * np.sin(angle_v)
                z_ = rcz + r * np.cos(angle_v) * np.sin(angle_h)
                if y_ > np.sin(x_)*np.cos(z_)*0.8 + np.sin(x_*0.5)*0.5: # Above ground
                    pts.append([x_, y_, z_])
                    # Gray rock color
                    shade = 100 + np.random.randint(60)
                    cols.append([shade, shade, shade+10])

    render_scene(np.array(pts, dtype=np.float64), 
                 np.array(cols, dtype=np.float64), 
                 output_dir, cam_distance=14)


def generate_mechanical(output_dir):
    """Example 3: An industrial mechanical part (Torus + Cylinder)"""
    pts, cols = [], []
    
    # Main Torus (Ring)
    R, r = 2.5, 0.8
    for theta in np.linspace(0, 2*np.pi, 200):
        for phi in np.linspace(0, 2*np.pi, 50):
            x_ = (R + r * np.cos(phi)) * np.cos(theta)
            y_ = r * np.sin(phi)
            z_ = (R + r * np.cos(phi)) * np.sin(theta)
            pts.append([x_, y_, z_])
            # Metallic blue / silver stripes
            if int(math.degrees(theta)) % 40 < 20:
                cols.append([180, 190, 210]) # Silver
            else:
                cols.append([60, 80, 180])   # Blue
                
    # Central Axle (Cylinder)
    axle_r = 0.5
    for angle in np.linspace(0, 2*np.pi, 40):
        for y_ in np.linspace(-4, 4, 80):
            x_ = axle_r * np.cos(angle)
            z_ = axle_r * np.sin(angle)
            pts.append([x_, y_, z_])
            shade = 140 + int(40 * np.cos(angle))
            cols.append([shade, shade, 60]) # Yellow metallic
            
    # End Caps (Discs)
    for y_ in [-4, 4]:
        for angle in np.linspace(0, 2*np.pi, 40):
            for radius in np.linspace(0, 1.5, 15):
                x_ = radius * np.cos(angle)
                z_ = radius * np.sin(angle)
                pts.append([x_, y_, z_])
                cols.append([180, 50, 50]) # Red warning caps

    render_scene(np.array(pts, dtype=np.float64), 
                 np.array(cols, dtype=np.float64), 
                 output_dir, cam_distance=14)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔄 Generating 3 Photogrammetry Example Datasets...")
    print("─────────────────────────────────────────────────")
    
    base_path = os.path.join(os.path.dirname(__file__), "Photogrammetry_Examples")
    
    print("\n🏛️  Generating Example 1: Temple")
    generate_temple(os.path.join(base_path, "1_Temple"))
    
    print("\n⛰️  Generating Example 2: Terrain")
    generate_terrain(os.path.join(base_path, "2_Terrain"))
    
    print("\n⚙️  Generating Example 3: Mechanical Part")
    generate_mechanical(os.path.join(base_path, "3_Mechanical"))
    
    print("\n─────────────────────────────────────────────────")
    print("✨ All examples generated successfully!")
    print(f"📁 Find them in: {os.path.abspath(base_path)}")
    print("\n💡 To use in the app: Click '📂 Load Images' and select one of the folders.")
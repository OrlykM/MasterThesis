import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def create_grid_floor(size=20, n=20, z_height=0.0):
    lines, points = [], []
    for i in np.linspace(-size, size, n):
        points.append([i, -size, z_height])
        points.append([i, size, z_height])
        lines.append([len(points)-2, len(points)-1])
        points.append([-size, i, z_height])
        points.append([size, i, z_height])
        lines.append([len(points)-2, len(points)-1])
        
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.4, 0.4, 0.4] for _ in range(len(lines))])
    return grid

def generate_dark_legend_image():
    bg_color = '#1a1a26' 
    fig, ax = plt.subplots(figsize=(1.5, 4), facecolor=bg_color)    
    fig.subplots_adjust(left=0.1, right=0.35)

    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=0.5, vmax=1.0)
    
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    
    cb.set_label('Occupancy Prob.', color='white', fontsize=12, weight='bold', labelpad=15)
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    legend_path = "legend_ui.png"
    plt.savefig(legend_path, dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return legend_path

def visualize_npz(npz_path, threshold=0.0):
    print(f"Завантаження карти: {npz_path}...")
    data = np.load(npz_path)
    log_odds, origin, resolution = data['log_odds'], data['origin'], data['resolution']
    
    occupied_indices = np.argwhere(log_odds > threshold)
    if len(occupied_indices) == 0:
        print("Немає зайнятих вокселів!")
        return

    probabilities = 1.0 / (1.0 + np.exp(-log_odds[log_odds > threshold]))
    norm_probs = np.clip((probabilities - 0.5) / 0.5, 0, 1)

    colors = plt.get_cmap("plasma")(norm_probs)[:, :3]
    points_world = origin + (occupied_indices * resolution) + (resolution / 2.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=resolution*3, max_nn=30))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    min_z = np.min(points_world[:, 2]) - 1.0 
    max_xy = np.max(np.abs(points_world[:, :2])) + 5.0
    floor_grid = create_grid_floor(size=max_xy, n=int(max_xy), z_height=min_z)

    legend_img_path = generate_dark_legend_image()

    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(f"Probability Heatmap: {npz_path}", 1280, 720)
    
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.1, 0.1, 0.15, 1.0]) 
    
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultLit"
    mat_pcd.point_size = 3.0
    
    mat_lines = rendering.MaterialRecord()
    mat_lines.shader = "unlitLine"
    mat_lines.line_width = 2.0

    scene_widget.scene.add_geometry("pcd", pcd, mat_pcd)
    scene_widget.scene.add_geometry("grid", floor_grid, mat_lines)
    scene_widget.scene.add_geometry("axes", axes, mat_lines)
    
    bounds = pcd.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    o3d_img = o3d.io.read_image(legend_img_path)
    img_widget = gui.ImageWidget(o3d_img)
    
    window.add_child(scene_widget)
    window.add_child(img_widget)
    
    def on_layout(layout_context):
        r = window.content_rect
        scene_widget.frame = r
        pref = img_widget.calc_preferred_size(layout_context, gui.Widget.Constraints())
        img_widget.frame = gui.Rect(r.width - pref.width - 20, r.height - pref.height - 20, pref.width, pref.height)
        
    window.set_on_layout(on_layout)
    
    print("Відкриття 3D вікна...")
    gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("map_file")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    
    visualize_npz(args.map_file, args.threshold)
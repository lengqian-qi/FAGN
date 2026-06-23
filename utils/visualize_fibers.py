import os
import json
import numpy as np
import nibabel as nib
import h5py
from scipy.interpolate import interp1d
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_selected_fibers_error(
        features,
        labels,
        preds,
        expected_tracts,
        expected_tract_selects,
        output_dir="fiber_visualization"
):
    """
    Parameters
    ----------
    features : (N, P, 3)
    labels   : (N,)
    preds    : (N,)
    expected_tracts : list
        all tracts
    expected_tract_selects : list
    """

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # tract color
    # =====================================================
    tract_color_map = {
        'AF': '#D55E5E',
        'FX': '#4C72B0',
        'SLF_III': '#55A868',
        'ILF': '#8172B2',
        'SLF_I': '#CCB974',  
        'FPT': '#D68B5C'      
    }

    # =====================================================
    # global coordinates
    # =====================================================
    all_coords = features.reshape(-1, 3)

    center_x = (all_coords[:, 0].min() + all_coords[:, 0].max()) / 2
    center_y = (all_coords[:, 1].min() + all_coords[:, 1].max()) / 2
    center_z = (all_coords[:, 2].min() + all_coords[:, 2].max()) / 2

    max_range = max(
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
        all_coords[:, 2].max() - all_coords[:, 2].min()
    )

    # =====================================================
    # Iterate through the fibers to be displayed
    # =====================================================
    for tract_name in expected_tract_selects:

        # -------------------------------------------------
        # matches left and right
        # -------------------------------------------------
        matched_indices = []
        matched_names = []

        for idx, full_name in enumerate(expected_tracts):

            if (
                full_name == tract_name
                or full_name.startswith(f"{tract_name}_")
                or tract_name in full_name
            ):
                matched_indices.append(idx)
                matched_names.append(full_name)

        if len(matched_indices) == 0:
            print(f"Warning: {tract_name} not found")
            continue

        print("\n" + "=" * 60)
        print(f"Processing: {tract_name}")
        print(f"Matched tracts: {matched_names}")

        # -------------------------------------------------
        # color
        # -------------------------------------------------
        highlight_color = '#FF4D4D'

        for tract_key in tract_color_map:
            if tract_key in tract_name:
                highlight_color = tract_color_map[tract_key]
                break

        # =================================================
        # GT Fiber Map
        # =================================================
        fig = plt.figure(figsize=(10, 6), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_pane_color((0, 0, 0, 0))
        ax.yaxis.set_pane_color((0, 0, 0, 0))
        ax.zaxis.set_pane_color((0, 0, 0, 0))

        ax.view_init(elev=20, azim=45)

        # -------------------------------------------------
        # all fibers in this tract
        # -------------------------------------------------
        selected_indices = np.where(
            np.isin(preds, matched_indices)
        )[0]

        print(f"Total fibers: {len(selected_indices)}")

        for idx in selected_indices:

            fiber = features[idx]

            ax.plot(
                fiber[:, 0],
                fiber[:, 1],
                fiber[:, 2],
                color=highlight_color,
                alpha=0.3,
                linewidth=0.5
            )

        ax.set_xlim(center_x - max_range / 2,
                    center_x + max_range / 2)

        ax.set_ylim(center_y - max_range / 2,
                    center_y + max_range / 2)

        ax.set_zlim(center_z - max_range / 2,
                    center_z + max_range / 2)

        legend_element = plt.Line2D(
            [0],
            [0],
            color=highlight_color,
            lw=2,
            label=tract_name
        )

        legend = ax.legend(
            handles=[legend_element],
            loc='upper left',
            fontsize=8,
            framealpha=0.3
        )

        for text in legend.get_texts():
            text.set_color('white')

        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_edgecolor('white')

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"{tract_name}.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            facecolor='black',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

        print(f"Saved: {save_path}")

        # =================================================
        # Error Map
        # =================================================
        error_mask = (
            np.isin(preds, matched_indices)
            &
            (preds != labels)
        )

        error_indices = np.where(error_mask)[0]

        print(
            f"Misclassified fibers: {len(error_indices)}"
        )

        if len(error_indices) == 0:
            continue

        fig = plt.figure(figsize=(10, 6), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_pane_color((0, 0, 0, 0))
        ax.yaxis.set_pane_color((0, 0, 0, 0))
        ax.zaxis.set_pane_color((0, 0, 0, 0))

        ax.view_init(elev=20, azim=45)

        for idx in error_indices:

            fiber = features[idx]

            ax.plot(
                fiber[:, 0],
                fiber[:, 1],
                fiber[:, 2],
                color=highlight_color,
                alpha=0.6,
                linewidth=0.5
            )

        ax.set_xlim(center_x - max_range / 2,
                    center_x + max_range / 2)

        ax.set_ylim(center_y - max_range / 2,
                    center_y + max_range / 2)

        ax.set_zlim(center_z - max_range / 2,
                    center_z + max_range / 2)

        legend_element = plt.Line2D(
            [0],
            [0],
            color=highlight_color,
            lw=2,
            label=f"{tract_name} Error"
        )

        legend = ax.legend(
            handles=[legend_element],
            loc='upper left',
            fontsize=8,
            framealpha=0.3
        )

        for text in legend.get_texts():
            text.set_color('white')

        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_edgecolor('white')

        plt.tight_layout()

        error_save_path = os.path.join(
            output_dir,
            f"{tract_name}_error.png"
        )

        plt.savefig(
            error_save_path,
            dpi=300,
            facecolor='black',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

        print(f"Saved: {error_save_path}")
        
        
        # =================================================
        # Error Map + GT Overlay
        # =================================================
        print(
            f"Overlay Error fibers: {len(error_indices)}"
        )

        fig = plt.figure(figsize=(10, 6), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_pane_color((0, 0, 0, 0))
        ax.yaxis.set_pane_color((0, 0, 0, 0))
        ax.zaxis.set_pane_color((0, 0, 0, 0))

        ax.view_init(elev=20, azim=45)

        # ==========================================
        # GT（gray）
        # ==========================================
        for idx in selected_indices:

            fiber = features[idx]

            ax.plot(
                fiber[:, 0],
                fiber[:, 1],
                fiber[:, 2],
                color='lightgray',
                alpha=0.3,
                linewidth=0.5
            )

        # ==========================================
        # Error（red）
        # ==========================================
        for idx in error_indices:

            fiber = features[idx]

            ax.plot(
                fiber[:, 0],
                fiber[:, 1],
                fiber[:, 2],
                color=highlight_color,
                alpha=0.6,
                linewidth=0.5
            )

        ax.set_xlim(
            center_x - max_range / 2,
            center_x + max_range / 2
        )

        ax.set_ylim(
            center_y - max_range / 2,
            center_y + max_range / 2
        )

        ax.set_zlim(
            center_z - max_range / 2,
            center_z + max_range / 2
        )

        # ==========================================
        # Legend
        # ==========================================
        gt_legend = plt.Line2D(
            [0],
            [0],
            color='lightgray',
            lw=2,
            label='GT'
        )

        error_legend = plt.Line2D(
            [0],
            [0],
            color=highlight_color,
            lw=2,
            label='Error'
        )

        legend = ax.legend(
            handles=[
                gt_legend,
                error_legend
            ],
            loc='upper left',
            fontsize=8,
            framealpha=0.3
        )

        for text in legend.get_texts():
            text.set_color('white')

        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_edgecolor('white')

        plt.tight_layout()

        overlay_save_path = os.path.join(
            output_dir,
            f"{tract_name}_error_with_GT.png"
        )

        plt.savefig(
            overlay_save_path,
            dpi=300,
            facecolor='black',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

        print(f"Saved: {overlay_save_path}")
    
def setup_axis(ax):

    ax.set_facecolor('black')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('off')

    ax.set_aspect('equal')
    
def plot_fibers_2d(
        ax,
        fibers,
        color,
        alpha=0.1,
        linewidth=0.5,
        view='xy'
):
    """
    view:
        xy
        yz
        xz
    """

    for fiber in fibers:

        if view == 'xy':
            ax.plot(
                fiber[:, 0],
                fiber[:, 1],
                color=color,
                alpha=alpha,
                linewidth=linewidth
            )

        elif view == 'yz':
            ax.plot(
                fiber[:, 1],
                fiber[:, 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth
            )

        elif view == 'xz':
            ax.plot(
                fiber[:, 0],
                fiber[:, 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth
            )
            
def plot_selected_fibers_multi_views_error(
        features,
        labels,
        preds,
        expected_tracts,
        expected_tract_selects,
        output_dir="fiber_visualization"
):

    os.makedirs(output_dir, exist_ok=True)

    tract_color_map = {
        'AF': '#D55E5E',
        'FX': '#4C72B0',
        'SLF_III': '#55A868',
        'ILF': '#8172B2',
        'SLF_I': '#CCB974',  
        'FPT': '#D68B5C'      
    }
    views = ['xy', 'yz', 'xz']

    for tract_name in expected_tract_selects:

        # =====================================================
        # left and right match
        # =====================================================
        matched_indices = []
        matched_names = []

        for idx, full_name in enumerate(expected_tracts):

            if (
                    full_name == tract_name
                    or full_name.startswith(f"{tract_name}_")
                    or tract_name in full_name
            ):
                matched_indices.append(idx)
                matched_names.append(full_name)

        if len(matched_indices) == 0:
            print(f"Warning: {tract_name} not found")
            continue

        print(f"\nProcessing {tract_name}")
        print(f"Matched: {matched_names}")

        # =====================================================
        # tract color
        # =====================================================
        highlight_color = '#FF4D4D'

        for tract_key in tract_color_map:
            if tract_key in tract_name:
                highlight_color = tract_color_map[tract_key]
                break

        # =====================================================
        # GT fibers
        # =====================================================
        selected_indices = np.where(
            np.isin(preds, matched_indices)
        )[0]

        selected_fibers = features[selected_indices]

        # =====================================================
        # Error fibers (FN)
        # =====================================================
        error_indices = np.where(
            np.isin(preds, matched_indices)
            &
            (preds != labels)
        )[0]

        error_fibers = features[error_indices]

        print(
            f"GT fibers={len(selected_indices)} "
            f"Error fibers={len(error_indices)}"
        )

        # =====================================================
        # 3 views
        # =====================================================
        for view in views:

            # =================================================
            # 1. GT
            # =================================================
            fig, ax = plt.subplots(
                figsize=(6, 6),
                facecolor='black'
            )

            setup_axis(ax)

            plot_fibers_2d(
                ax,
                selected_fibers,
                color=highlight_color,
                alpha=0.3,
                linewidth=0.5,
                view=view
            )

            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                f"{tract_name}_{view}.png"
            )

            plt.savefig(
                save_path,
                dpi=300,
                facecolor='black',
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close()

            # =================================================
            # 2. Error
            # =================================================
            fig, ax = plt.subplots(
                figsize=(6, 6),
                facecolor='black'
            )

            setup_axis(ax)

            plot_fibers_2d(
                ax,
                error_fibers,
                color=highlight_color,
                alpha=0.6,
                linewidth=0.5,
                view=view
            )

            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                f"{tract_name}_error_{view}.png"
            )

            plt.savefig(
                save_path,
                dpi=300,
                facecolor='black',
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close()

            # =================================================
            # 3. Overlay
            # =================================================
            fig, ax = plt.subplots(
                figsize=(6, 6),
                facecolor='black'
            )

            setup_axis(ax)

            # GT
            plot_fibers_2d(
                ax,
                selected_fibers,
                color='lightgray',
                alpha=0.3,
                linewidth=0.5,
                view=view
            )

            # Error
            plot_fibers_2d(
                ax,
                error_fibers,
                color=highlight_color,
                alpha=0.6,
                linewidth=0.5,
                view=view
            )

            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                f"{tract_name}_error_with_GT_{view}.png"
            )

            plt.savefig(
                save_path,
                dpi=300,
                facecolor='black',
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close()

            print(
                f"Saved "
                f"{tract_name}_{view}"
            )
            
if __name__ == "__main__":
    features_h5_path = ""
    labels_h5_path = ""

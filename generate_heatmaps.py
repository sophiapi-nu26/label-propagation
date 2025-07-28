#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_csv_as_array(csv_path):
    """
    Load a CSV file of shape (GRID_SIZE+1, GRID_SIZE+1) into a NumPy array.
    """
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            # Split on commas, convert to float, ignoring empty strings
            row = [float(x) for x in line.strip().split(',') if x != '']
            data.append(row)
    return np.array(data)


def plot_single_heatmap(data, outpath, title=None):
    """
    Render a single 2D heatmap with matplotlib, including:
      - Tick marks + axis labels
      - A color bar labeled 'Value'
      - An optional title at the top
    Then save to outpath as a PNG (no subplots).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Force the color scale to be [0..1]
    im = ax.imshow(data, origin='lower', vmin=0.0, vmax=1.0)

    # Add a color bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    # Set up tick marks
    shape_y, shape_x = data.shape
    num_ticks = 5  
    x_positions = np.linspace(0, shape_x - 1, num_ticks)
    y_positions = np.linspace(0, shape_y - 1, num_ticks)

    ax.set_xticks(x_positions)
    ax.set_yticks(y_positions)
    ax.set_xticklabels([f"{int(v)}" for v in x_positions])
    ax.set_yticklabels([f"{int(v)}" for v in y_positions])

    # Axis labels
    ax.set_xlabel("Beta Index")
    ax.set_ylabel("Alpha Index")

    if title:
        ax.set_title(title)

    plt.savefig(outpath, bbox_inches='tight')
    plt.close(fig)



def stitch_images(image_paths, outpath, mode='2h'):
    """
    Combine multiple images into one final image using PIL.
    mode='2h': 2 images horizontally (side by side).
    mode='2v': 2 images vertically (stacked).
    mode='2x2': 4 images in a 2x2 grid.

    Saves the stitched result to outpath (PNG), then removes the individual images.
    """
    imgs = [Image.open(p) for p in image_paths]

    if mode == '2h':
        # Exactly 2 images side by side
        w1, h1 = imgs[0].size
        w2, h2 = imgs[1].size
        out_w = w1 + w2
        out_h = max(h1, h2)
        new_img = Image.new('RGB', (out_w, out_h))
        new_img.paste(imgs[0], (0, 0))
        new_img.paste(imgs[1], (w1, 0))

    elif mode == '2v':
        # 2 images vertically
        w1, h1 = imgs[0].size
        w2, h2 = imgs[1].size
        out_w = max(w1, w2)
        out_h = h1 + h2
        new_img = Image.new('RGB', (out_w, out_h))
        new_img.paste(imgs[0], (0, 0))
        new_img.paste(imgs[1], (0, h1))

    elif mode == '2x2':
        # 4 images in a 2x2 arrangement: [0,1; 2,3]
        w, h = imgs[0].size
        # (We assume all 4 images have the same size.)
        new_img = Image.new('RGB', (2 * w, 2 * h))
        new_img.paste(imgs[0], (0,   0  ))
        new_img.paste(imgs[1], (w,   0  ))
        new_img.paste(imgs[2], (0,   h  ))
        new_img.paste(imgs[3], (w,   h  ))
    else:
        raise ValueError("Invalid stitch mode.")

    new_img.save(outpath)

    # Remove temp images
    for p in image_paths:
        os.remove(p)


def gather_csv_files(subfolder, prefix):
    """
    Gather (round_idx, path) pairs for CSV files that match the naming pattern
    in the subfolder.
    Example: prefix="conv_smallest_in_comm_c0_round_" might match files
      "conv_smallest_in_comm_c0_round_0.csv", "conv_smallest_in_comm_c0_round_1.csv", etc.
    Returns a sorted list of (round_idx, fullpath) by ascending round_idx.
    """
    data_files = []
    for fname in os.listdir(subfolder):
        if fname.startswith(prefix) and fname.endswith(".csv"):
            # parse the round from the filename
            # e.g. "conv_smallest_in_comm_c0_round_3.csv"
            parts = fname.split("_")
            # last part might be "3.csv"
            round_part = parts[-1]
            round_str = round_part.replace(".csv", "")
            round_idx = int(round_str)
            fullpath = os.path.join(subfolder, fname)
            data_files.append((round_idx, fullpath))
    # sort by round_idx
    data_files.sort(key=lambda x: x[0])
    return data_files


def generate_heatmaps_for_run(run_folder):
    """
    Given the path to a run folder (e.g. "./run_202547_19224"),
    we create for each statistic and for each round a composite image containing:
      - 2 sub-images for c0/c1-type stats
      - 4 sub-images for cross_label_distribution
    Each sub-image is a single heatmap with colorbar, axis ticks, labels, etc.
    """
    # Identify subfolders
    csc_path = os.path.join(run_folder, "conv_smallest_in_comm")
    csg_path = os.path.join(run_folder, "conv_smallest_global")
    fnc_path = os.path.join(run_folder, "fraction_not_changed")
    cld_path = os.path.join(run_folder, "cross_label_distribution")

    # A) conv_smallest_in_comm => c0, c1
    csc0 = gather_csv_files(csc_path, "conv_smallest_in_comm_c0_round_")
    csc1 = gather_csv_files(csc_path, "conv_smallest_in_comm_c1_round_")

    for (round_idx, file_c0), (_, file_c1) in zip(csc0, csc1):
        arr_c0 = load_csv_as_array(file_c0)
        arr_c1 = load_csv_as_array(file_c1)

        temp_c0 = os.path.join(csc_path, f"temp_c0_{round_idx}.png")
        temp_c1 = os.path.join(csc_path, f"temp_c1_{round_idx}.png")

        plot_single_heatmap(arr_c0, temp_c0, title=f"c0 (round={round_idx})")
        plot_single_heatmap(arr_c1, temp_c1, title=f"c1 (round={round_idx})")

        final_out = os.path.join(csc_path, f"conv_smallest_in_comm_round_{round_idx}.png")
        stitch_images([temp_c0, temp_c1], final_out, mode='2h')
        print(f"Saved {final_out}")

    # B) conv_smallest_global => c0, c1
    csg0 = gather_csv_files(csg_path, "conv_smallest_global_c0_round_")
    csg1 = gather_csv_files(csg_path, "conv_smallest_global_c1_round_")

    for (round_idx, file_c0), (_, file_c1) in zip(csg0, csg1):
        arr_c0 = load_csv_as_array(file_c0)
        arr_c1 = load_csv_as_array(file_c1)

        temp_c0 = os.path.join(csg_path, f"temp_c0_{round_idx}.png")
        temp_c1 = os.path.join(csg_path, f"temp_c1_{round_idx}.png")

        plot_single_heatmap(arr_c0, temp_c0, title=f"c0 (round={round_idx})")
        plot_single_heatmap(arr_c1, temp_c1, title=f"c1 (round={round_idx})")

        final_out = os.path.join(csg_path, f"conv_smallest_global_round_{round_idx}.png")
        stitch_images([temp_c0, temp_c1], final_out, mode='2h')
        print(f"Saved {final_out}")

    # C) fraction_not_changed => c0, c1
    fnc0 = gather_csv_files(fnc_path, "fraction_not_changed_c0_round_")
    fnc1 = gather_csv_files(fnc_path, "fraction_not_changed_c1_round_")

    for (round_idx, file_c0), (_, file_c1) in zip(fnc0, fnc1):
        arr_c0 = load_csv_as_array(file_c0)
        arr_c1 = load_csv_as_array(file_c1)

        temp_c0 = os.path.join(fnc_path, f"temp_c0_{round_idx}.png")
        temp_c1 = os.path.join(fnc_path, f"temp_c1_{round_idx}.png")

        plot_single_heatmap(arr_c0, temp_c0, title=f"c0 (round={round_idx})")
        plot_single_heatmap(arr_c1, temp_c1, title=f"c1 (round={round_idx})")

        final_out = os.path.join(fnc_path, f"fraction_not_changed_round_{round_idx}.png")
        stitch_images([temp_c0, temp_c1], final_out, mode='2h')
        print(f"Saved {final_out}")

    # D) cross_label_distribution => 00, 01, 10, 11
    cld00 = gather_csv_files(cld_path, "cross_label_dist_00_round_")
    cld01 = gather_csv_files(cld_path, "cross_label_dist_01_round_")
    cld10 = gather_csv_files(cld_path, "cross_label_dist_10_round_")
    cld11 = gather_csv_files(cld_path, "cross_label_dist_11_round_")
    
    

    for ((round_idx, f00), (_, f01), (_, f10), (_, f11)) in zip(cld00, cld01, cld10, cld11):
        arr_00 = load_csv_as_array(f00)
        arr_01 = load_csv_as_array(f01)
        arr_10 = load_csv_as_array(f10)
        arr_11 = load_csv_as_array(f11)

        t00 = os.path.join(cld_path, f"temp_00_{round_idx}.png")
        t01 = os.path.join(cld_path, f"temp_01_{round_idx}.png")
        t10 = os.path.join(cld_path, f"temp_10_{round_idx}.png")
        t11 = os.path.join(cld_path, f"temp_11_{round_idx}.png")

        plot_single_heatmap(arr_00, t00, title=f"Comm 0 nodes with initial min label from Comm 0 (round {round_idx})")
        plot_single_heatmap(arr_01, t01, title=f"Comm 0 nodes with initial min label from Comm 1 (round {round_idx})")
        plot_single_heatmap(arr_10, t10, title=f"Comm 1 nodes with initial min label from Comm 0 (round {round_idx})")
        plot_single_heatmap(arr_11, t11, title=f"Comm 1 nodes with initial min label from Comm 1 (round {round_idx})")

        final_out = os.path.join(cld_path, f"cross_label_dist_round_{round_idx}.png")
        stitch_images([t00, t01, t10, t11], final_out, mode='2x2')
        print(f"Saved {final_out}")

    print("All heatmaps generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps with color bars, axis labels, etc. from a label-propagation results folder."
    )
    parser.add_argument(
        "--run_folder", type=str, required=True,
        help="Name of the run folder, e.g. './run_202547_19224'."
    )
    args = parser.parse_args()

    run_folder = args.run_folder
    if not os.path.isdir(run_folder):
        print(f"Error: folder {run_folder} does not exist.")
        return

    generate_heatmaps_for_run(run_folder)


if __name__ == "__main__":
    main()

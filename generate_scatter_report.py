from docx import Document
from docx.shared import Inches
import os


def generate_scatter_report(plots_dir, output_path, dataset_type="test"):
    """Generate a DOCX report with scatter plots arranged in a table.

    Parameters
    ----------
    plots_dir : str
        Directory containing subfolders for each farm. Each subfolder should
        include images named ``scatter_day1.png``, etc.
    output_path : str
        Path to the output DOCX file.
    dataset_type : str, optional
        One of "test", "train", "val". Determines which scatter plots to include.
    """
    if not os.path.exists(plots_dir):
        raise FileNotFoundError(f"Directory '{plots_dir}' not found")

    document = Document()
    document.add_heading(f"{dataset_type.capitalize()} Forecast Scatter Plots", level=1)

    farms = [
        d for d in os.listdir(plots_dir) if os.path.isdir(os.path.join(plots_dir, d))
    ]
    farms.sort()

    table = document.add_table(rows=1 + len(farms), cols=4)
    table.style = "Light List"

    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Farm"
    hdr_cells[1].text = "Day +1"
    hdr_cells[2].text = "Day +2"
    hdr_cells[3].text = "Day +3"

    for row_idx, farm in enumerate(farms, start=1):
        row_cells = table.rows[row_idx].cells
        row_cells[0].text = farm.replace("_", " ")
        for day in range(1, 4):
            if dataset_type == "test":
                img_name = f"scatter_day{day}.png"
            else:
                img_name = f"{dataset_type}_scatter_day{day}.png"
            img_path = os.path.join(plots_dir, farm, img_name)
            if os.path.isfile(img_path):
                paragraph = row_cells[day].paragraphs[0]
                run = paragraph.add_run()
                run.add_picture(img_path, width=Inches(2))
            else:
                row_cells[day].text = "N/A"

    document.save(output_path)
    print(f"Report saved to {output_path}")

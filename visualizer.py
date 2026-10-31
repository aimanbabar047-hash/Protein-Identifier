"""
visualizer.py
==============
Renders the Needleman-Wunsch score matrix as an HTML table, with the
traceback path highlighted -- similar to how it's drawn by hand on
paper (grid + arrows), to help beginners see how the algorithm works.
"""


def get_traceback_path(traceback_matrix, n, m):
    """
    Walks the traceback matrix from the bottom-right corner to the
    top-left, returning the list of (i, j) cells that make up the
    optimal alignment path.
    """
    path = [(n, m)]
    i, j = n, m

    while i > 0 or j > 0:
        direction = traceback_matrix[i][j]
        if direction == 'D':
            i -= 1
            j -= 1
        elif direction == 'U':
            i -= 1
        else:  # 'L'
            j -= 1
        path.append((i, j))

    return set(path)


# Small arrow symbols to show direction inside each cell
DIRECTION_ARROWS = {
    'D': '↖',
    'U': '↑',
    'L': '←',
    '': ''
}


def render_matrix_html(seq1, seq2, score_matrix, traceback_matrix):
    """
    Builds an HTML table representing the score matrix, with:
      - row/column headers showing the sequence letters
      - a small arrow in each cell showing its traceback direction
      - the optimal traceback path highlighted in a different color
    """
    n, m = len(seq1), len(seq2)
    path_cells = get_traceback_path(traceback_matrix, n, m)

    row_labels = ['-'] + list(seq1)
    col_labels = ['-'] + list(seq2)

    html = """
    <style>
        .nw-table {
            border-collapse: separate;
            border-spacing: 3px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        .nw-table th, .nw-table td {
            border: 1px solid #d8dee9;
            width: 48px; height: 40px;
            text-align: center; vertical-align: middle;
            border-radius: 6px;
        }
        .nw-table th {
            background-color: #1a2b4c;
            color: #ffffff;
            font-weight: 600;
        }
        .nw-header-corner {
            background-color: #1a2b4c;
        }
        .nw-table td {
            background-color: #ffffff;
        }
        .nw-path {
            background-color: #ff6b6b !important;
            color: #ffffff;
            font-weight: 700;
            border: 1px solid #e04b4b !important;
        }
        .nw-arrow {
            font-size: 11px;
            color: #9aa5b1;
            display: block;
        }
        .nw-path .nw-arrow {
            color: #ffe8e8;
        }
        .nw-score { font-size: 14px; }
    </style>
    <table class="nw-table">
    """

    # Header row (col labels)
    html += "<tr><th class='nw-header-corner'></th>"
    for col in col_labels:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # Data rows
    for i, row_label in enumerate(row_labels):
        html += f"<tr><th>{row_label}</th>"
        for j in range(len(col_labels)):
            score = score_matrix[i][j]
            direction = traceback_matrix[i][j]
            arrow = DIRECTION_ARROWS.get(direction, '')

            cell_class = "nw-path" if (i, j) in path_cells else ""
            html += (
                f"<td class='{cell_class}'>"
                f"<span class='nw-arrow'>{arrow}</span>"
                f"<span class='nw-score'>{score}</span>"
                f"</td>"
            )
        html += "</tr>"

    html += "</table>"
    return html

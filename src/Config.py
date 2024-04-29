import os
import git

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #
# ----------------------------------- DASH ----------------------------------- #
debug = True
port = 5000

# ---------------------------------- MODELS ---------------------------------- #
difficulty_estimation = True
simplification = False
quantization = False

# --------------------------------- INTERFACE -------------------------------- #
difficulty_colors = {
    "A1": "lime",
    "A2": "blue",
    "B1": "violet",
    "B2": "yellow",
    "C1": "orange",
    "C2": "red",
}

# ---------------------------------------------------------------------------- #
#                                    STATIC                                    #
# ---------------------------------------------------------------------------- #
pwd = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
title_length = 60
description_length = 300

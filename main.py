import os

# Set CUDA_VISIBLE_DEVICES=""
os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == "__main__":
    # # Load the models
    # import src.backend.Models as Models

    # Load the app
    import src.frontend.App as App

    App.run()

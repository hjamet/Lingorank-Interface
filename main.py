from src.frontend.app import App

if __name__ == "__main__":
    # Create the app
    app = App()

    # Run the server
    app(debug=True, port=5000)

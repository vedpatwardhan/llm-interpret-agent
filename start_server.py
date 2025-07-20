import sys
import time

sys.path.append("./circuit-tracer")
from circuit_tracer.frontend.local_server import serve


if __name__ == "__main__":
    port = 8046
    server = None  # Initialize server to None

    try:
        server = serve(data_dir='./circuit-tracer/graph_files/', port=port)
        print(f"Serving attribution graph on http://localhost:{port}")
        print("Press Ctrl+C to stop the server.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down server...")
        if server:
            server.stop()  # Call the stop method of your server object
            print("Server stopped gracefully.")
        else:
            print("Server was not started.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if server:
            server.stop()
    finally:
        print("Exiting script.")

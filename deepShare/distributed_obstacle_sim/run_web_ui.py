"""Run the Streamlit web UI for the distributed obstacle simulator.

Usage:
    streamlit run run_web_ui.py --server.address 0.0.0.0 --server.port 8501
"""

from simulator.ui.web_app import main


if __name__ == "__main__":
    main()

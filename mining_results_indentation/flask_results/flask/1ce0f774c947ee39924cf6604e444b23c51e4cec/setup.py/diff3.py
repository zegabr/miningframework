from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[
<<<<<<< /home/ze/miningframework/mining_results_indentation/flask_results/flask/1ce0f774c947ee39924cf6604e444b23c51e4cec/setup.py/left.py
        "Werkzeug>=2.0",
        "Jinja2>=3.0",
        "itsdangerous>=2.0",
        "click>=8.0",
=======
        "Werkzeug >= 2.0",
        "Jinja2 >= 3.0",
        "itsdangerous >= 2.0",
        "click >= 7.1.2", # CReduzido
>>>>>>> /home/ze/miningframework/mining_results_indentation/flask_results/flask/1ce0f774c947ee39924cf6604e444b23c51e4cec/setup.py/right.py
    ],
    extras_require={
        "async": ["asgiref >= 3.2"],
        "dotenv": ["python-dotenv"],
    },
)

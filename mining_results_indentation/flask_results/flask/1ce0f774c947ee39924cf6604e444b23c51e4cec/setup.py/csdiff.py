from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[
        "Werkzeug >= 2.0",
        "Jinja2 >= 3.0",
        "itsdangerous >= 2.0",
        "
<<<<<<< /home/ze/miningframework/mining_results/flask_results/flask/1ce0f774c947ee39924cf6604e444b23c51e4cec/setup.py/left.py
click>=8.0
=======
click >= 7.1.2
>>>>>>> /home/ze/miningframework/mining_results/flask_results/flask/1ce0f774c947ee39924cf6604e444b23c51e4cec/setup.py/right.py
",
    ],
    extras_require={
        "async": ["asgiref >= 3.2"],
        "dotenv": ["python-dotenv"],
    },
)

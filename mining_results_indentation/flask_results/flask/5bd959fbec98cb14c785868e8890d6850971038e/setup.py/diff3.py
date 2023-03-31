from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[ # CReduzido
<<<<<<< /home/ze/miningframework/mining_results/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
        "Werkzeug>=2.0",
        "Jinja2>=3.0",
        "itsdangerous>=2.0",
        "click>=7.1.2",
=======
        "Werkzeug >= 0.15, < 2.0",
        "Jinja2 >= 2.10.1, < 3.0",
        "itsdangerous >= 0.24, < 2.0",
        "click >= 5.1, < 8.0",
>>>>>>> /home/ze/miningframework/mining_results/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
    ],
    extras_require={
        "async": ["asgiref>=3.2"],
        "dotenv": ["python-dotenv"],
    },
)
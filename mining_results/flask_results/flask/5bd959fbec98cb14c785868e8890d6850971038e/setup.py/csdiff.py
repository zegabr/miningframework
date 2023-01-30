from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[
        "Werkzeug>=2.0",
        "Jinja2>=3.0",
        "itsdangerous>=2.0",
        "click>=7.1.2",
    ],
    extras_require={
        "async": ["asgiref>=3.2"],
<<<<<<< temp/left.py

=======
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=[
        "Werkzeug >= 0.15, < 2.0",
        "Jinja2 >= 2.10.1, < 3.0",
        "itsdangerous >= 0.24, < 2.0",
        "click >= 5.1, < 8.0",
    ],
    extras_require={

>>>>>>> temp/right.py
        "dotenv": ["python-dotenv"],
    },
)

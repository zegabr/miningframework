from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[
<<<<<<< ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
        "Werkzeug>=2.0"
=======
        "Werkzeug >= 0.15
>>>>>>> ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
, < 2.0",
<<<<<<< ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
        "Jinja2>=3.0"
=======
        "Jinja2 >= 2.10.1
>>>>>>> ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
, < 3.0",
<<<<<<< ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
        "itsdangerous>=2.0"
=======
        "itsdangerous >= 0.24
>>>>>>> ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
, < 2.0",
<<<<<<< ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
        "click>=7.1.2"
=======
        "click >= 5.1
>>>>>>> ./flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
, < 8.0",
    ],
    extras_require={
        "async": ["asgiref>=3.2"],
        "dotenv": ["python-dotenv"],
    },
)

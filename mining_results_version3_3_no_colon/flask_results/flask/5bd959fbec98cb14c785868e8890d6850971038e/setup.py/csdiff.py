from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Flask",
    install_requires=[
        "
<<<<<<< /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
Werkzeug>=2.0
=======
Werkzeug >= 0.15, < 2.0
>>>>>>> /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
",
        "
<<<<<<< /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
Jinja2>=3.0
=======
Jinja2 >= 2.10.1, < 3.0
>>>>>>> /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
",
        "
<<<<<<< /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
itsdangerous>=2.0
=======
itsdangerous >= 0.24, < 2.0
>>>>>>> /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
",
        "
<<<<<<< /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/left.py
click>=7.1.2
=======
click >= 5.1, < 8.0
>>>>>>> /home/ze/miningframework/mining_results_version3_3_no_colon/flask_results/flask/5bd959fbec98cb14c785868e8890d6850971038e/setup.py/right.py
",
    ],
    extras_require={
        "async": ["asgiref>=3.2"],
        "dotenv": ["python-dotenv"],
    },
)

from setuptools import find_packages,setup
from typing import List
import os

def get_requirements()->List[str]:

    """
    This function will return the list of requirements
    """
    requirement_list:List[str]=[]
    path = os.path.abspath('requirements.txt')

    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line != '-e .' and line != '-e .\n':
            if line.endswith('\n'):
                requirement_list.append(line[:-1])
            else:
                requirement_list.append(line)

    return requirement_list


setup(
    name = "Student_Performance",
    version = "0.0.1",
    author = "Subhasish",
    author_email = "subhasishsaha007@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)

# python setup.py
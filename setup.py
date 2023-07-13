from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements.
    '''
    try:
        requirements = []
        with open(file_path) as f:
            requirements = f.readlines()
            requirements = [req.replace("\n","") for req in requirements]

            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)

        return requirements
    
    except Exception as e:
        print(f'Error : {e}')

    setup(
        name='project2',
        version='0.0.1',
        author='Subhasish-Saha',
        author_email='subhasishsaha007@gmail.com',
        packages=find_packages(),
        install_requires = get_requirements('requirements.txt')
    )

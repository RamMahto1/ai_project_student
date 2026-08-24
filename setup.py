from setuptools import find_packages, setup
from typing import List

HYPEN_DOT_E = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    this function is return list of function
    '''

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.strip()for req in requirements]

        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)

        return requirements

setup(
    name="ai_project_student_profemance",
    version="0.0.1",
    author="ram",
    author_email="rammahto645@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)
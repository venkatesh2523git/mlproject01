from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        ## readlines create '\n' at the end of each library name so replace that with blank
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements: # '-e . ' should not get installed when this fun get triggered so we need to remove that from the list
            requirements.remove(HYPHEN_E_DOT)
    return requirements
        

setup(
    name='mlproject',
    version='0.0.1',
    author = 'Vicky',
    author_email='chinthavenkatesh2523@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),

) 
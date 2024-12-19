import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent.resolve()

PACKAGE_NAME = "vbavatar"
AUTHOR = "Shaked Zychlinski"
AUTHOR_EMAIL = "shakedzy@gmail.com"

LICENSE = "CC BY-NC 4.0"
DESCRIPTION = "LLM and VLM based avatar"
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf8")
LONG_DESC_TYPE = "text/markdown"

requirements = (HERE / "requirements.txt").read_text(encoding="utf8")
INSTALL_REQUIRES = [s.strip() for s in requirements.split("\n")]


setup(
    name=PACKAGE_NAME,
    version='0.0.0',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            f'vba = {PACKAGE_NAME}.__main__:run'
        ]
    }
)
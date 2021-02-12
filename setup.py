import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="smg-pyopenpose",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Interface to OpenPose",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-pyopenpose",
    packages=["smg.pyopenpose"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-contrib-python==3.4.2.16",
        "smg-opengl",
        "smg-skeletons"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.*',
)

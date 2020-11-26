import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thesne",
    version="0.0.1",
    author="ak0592",
    author_email="forgit8b.bb8@gmail.com",
    description="You can receive the message 'Hello!!!'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https:/github.com/ak0592/thesne.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_require=[
    	"numpy",
		"theano",
		"sklearn",
		"pandas"
	],
)
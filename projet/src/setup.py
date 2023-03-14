import setuptools

setuptools.setup(
	name = "INSA-IdKey",
	version ="0.0.1",
	author = "Simon Buré, Lionel Dalmau, Mayoran Raveendran, Olivia Seffacene, Jesus Uxue",
	author_email = "lionel.dalmau@insa-lyon.fr",
	description = "Test of project deployement",
	long_description = "Test of deployement for devloppement software project",
	long_description_content_type = "text/markdown",
	packages = setuptools.find_packages(),
	classifiers = [
	"Programming Language :: Python :: 3 ",
	"Operating System :: OS Independent",
	],
	install_requires = ["tk", "sqlite3", "torch", "sklearn"]
	)

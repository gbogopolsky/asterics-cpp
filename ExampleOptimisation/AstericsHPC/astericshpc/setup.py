'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

from setuptools import setup
from setuptools import Extension
from setuptools.dist import Distribution
from sys import prefix
import subprocess
import numpy as np

try:
	from Cython.Distutils import build_ext
except ImportError:
	use_cython = False
	print('Cython not found')
	raise Exception('Please install Cython on your system')
else:
	use_cython = True

NAME = 'astericshpc'
VERSION = '0.1'
AUTHOR = 'Pierre Aubert'
AUTHOR_EMAIL = 'aubertp7@gmail.com'
URL = ''
DESCRIPTION = 'Basic functions for ASTERICS HPC lecture'
LICENSE = 'CeCILL-C'

def get_prefix():
	"""
	Get prefix from either config file or command line
	:return: str
	prefix install path
	"""
	dist = Distribution()
	dist.parse_config_files()
	dist.parse_command_line()
	try:
		user_prefix = dist.get_option_dict('install')['prefix'][1]
	except KeyError:
		user_prefix = prefix
	return user_prefix

import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
	if type(value) == str:
		value = value.replace(" -Wstrict-prototypes ", " ")
		value = value.replace(" -DNDEBUG ", " ")
		value = value.replace(" -g ", " ")
		cfg_vars[key] = value

extra_compile_args = ['-Wno-invalid-offsetof']

def use_clang():
	if 'gcc' in subprocess.getoutput("echo $CC"):
		return False
	elif 'clang' in subprocess.getoutput("echo $CC"):
		return True
	else:
		if 'not found' in subprocess.getoutput("gcc --version") or 'clang'  in subprocess.getoutput("gcc --version"):
			return True
		else: 			
			return False

if use_clang():
	clangVersion = subprocess.getoutput("clang --version").split()
	i = 0
	while clangVersion[i] != "version":
		i += 1
	clangMainVersion = int(clangVersion[i + 1].split(".")[0])
	print("Find version of Clang ", clangMainVersion)
	if clangMainVersion > 9:
		extra_compile_args.append('-Wno-unused-command-line-argument') #no need for clang 10.0
		extra_compile_args.append("-Wno-injected-class-name")
		extra_compile_args.append("-Wno-macro-redefined")

packageName = 'astericshpc'
ext_modules = [
	Extension(packageName, ['@CMAKE_CURRENT_SOURCE_DIR@/astericshpc.cpp',
		'@CMAKE_CURRENT_SOURCE_DIR@/allocTableWrapper.cpp',
		'@CMAKE_CURRENT_SOURCE_DIR@/allocMatrixWrapper.cpp',
		'@CMAKE_CURRENT_SOURCE_DIR@/timerWrapper.cpp'
	],
	libraries=["asterics_hpc"],
	library_dirs=['@ASTERICS_CPP_LIBRARY_BUILD@'],
	runtime_library_dirs=['@ASTERICS_CPP_LIBRARY_DIR@'],
	extra_link_args=['-Wl,-rpath,@ASTERICS_CPP_LIBRARY_BUILD@'],
	extra_compile_args=extra_compile_args,

	include_dirs=['.',
		'@ASTERICS_HPC_INCLUDE@',
		np.get_include()]
	)
]

try:
	setup(name = NAME,
		version=VERSION,
		ext_modules=ext_modules,
		description=DESCRIPTION,
		install_requires=['numpy', 'cython'],
		author=AUTHOR,
		author_email=AUTHOR_EMAIL,
		license=LICENSE,
		url=URL,
		classifiers=[
		'Intended Audience :: Science/Research',
		'License :: OSI Approved ::Cecil-C',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
		'Topic :: Scientific/Engineering :: Astronomy',
		'Development Status :: 3 - Alpha'],
	)

except Exception as e:
	print(str(e))
	sys.exit(-1)


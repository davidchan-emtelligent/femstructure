from setuptools import setup, find_packages

setup(
    name='femstructure',
    version='0.1.0',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib'
    ],

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    entry_points={
        'console_scripts': [
            'create=femstructure.create_building.create:create_main',
            'remove=femstructure.create_building.create:remove_main',
            'plot=femstructure.create_building.plot:main',
            'frame=femstructure.main:frame_main',
            'truss=femstructure.main:truss_main'
        ],
    },

    package_data={
        'femstructure': ['create/config/*'],
    },
)


from setuptools import setup, find_packages

setup(
    name='GPTrip',
    version='1.0',
    description='Render trippy videos of Gaussian random fields dancing to music.',
    author='Kosio Karchev',
    author_email='kosiokarchev@gmail.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'gptrip = gptrip.cli.gptrip_cli:cli'
        ]
    },
    install_requires=[
        'numpy', 'matplotlib', 'ffmpeg-python', 'pydub'
    ],
    extras_require={
        'TORCH': 'torch'
    },
    python_requires='>=3.6',
    package_data={
        'colors': ['*.cmap']
    }
)

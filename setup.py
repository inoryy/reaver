import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='reaver',
    version='2.1.9',
    author='Roman Ring',
    author_email='inoryy@gmail.com',
    description='Reaver: Modular Deep Reinforcement Learning Framework. Focused on StarCraft II. '
                'Supports Gym, Atari, and MuJoCo. Matches reference results.',
    long_description=long_description,
    keywords='reaver starcraft2 gym atari mujoco tensorflow reinforcement learning neural network',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'PySC2 >= 3.0.0',
        'gin-config >= 0.3.0',
        'tensorflow >= 2.1.0',
        'tensorflow-probability >= 0.9.0'
    ],
    extras_require={
        'gym': [
            'PyOpenGL',
            'gym >= 0.9',
            'opencv-python'
        ],
        'atari': [
            'Pillow',
            'PyOpenGL',
            'gym >= 0.9',
            'atari_py >= 0.1.4',
        ],
        'mujoco': [
            'imageio',
            'gym >= 0.9',
            'mujoco_py >= 1.50',
        ]
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url='https://github.com/inoryy/reaver',
)

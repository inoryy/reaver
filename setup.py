from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='reaver',
    version='0.1',
    author='Roman Ring',
    author_email='inoryy@gmail.com',
    license='MIT',
    description='Deep Reinforcement Learning Agent for StarCraft II',
    long_description=long_description,
    url='https://github.com/inoryy/pysc2-rl-agent/',
    keywords='reaver starcraft tensorflow machine reinforcement learning neural network',
    packages=[
        'reaver',
        'reaver.env'
    ],
    install_requires=[
        'numpy',
        'PySC2>=2.0',
    ],
    extras_require={
        'tensorflow': [
            'tensorflow>=1.8.0',
            'tensorflow-probability>=0.4.0'
        ],
        'tensorflow with gpu': [
            'tensorflow-gpu>=1.8.0',
            'tensorflow-probability-gpu>=0.4.0'
        ],
    }
)

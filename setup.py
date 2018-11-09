from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='reaver',
    version='2.0',
    author='Roman Ring',
    author_email='inoryy@gmail.com',
    license='MIT',
    description='Deep Reinforcement Learning Agent for StarCraft II',
    long_description=long_description,
    url='https://github.com/inoryy/reaver/',
    keywords='reaver starcraft tensorflow machine reinforcement learning neural network',
    packages=[
        'reaver',
        'reaver.envs',
        'reaver.models',
        'reaver.agents'
    ],
    install_requires=[
        'numpy',
        'PySC2>=2.0',
    ],
    extras_require={
        'tf-cpu': [
            'tensorflow>=1.8.0',
            'tensorflow-probability>=0.4.0'
        ],
        'tf-gpu': [
            'tensorflow-gpu>=1.8.0',
            'tensorflow-probability-gpu>=0.4.0'
        ],
    }
)

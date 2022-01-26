from setuptools import setup, find_packages

setup(name='subgroupshift', 
      version='1.0', 
      author="Lisa Koch",
      author_email="lisa.koch@uni-tuebingen.de",      
      packages=find_packages(),
      install_requires=[
          'gdown',
          'matplotlib==3.5.0',
          'numpy==1.21.4',
          'pandas==1.3.4',
          'PyYAML==6.0',
          'scikit_learn==1.0.2',
          'scipy==1.7.3',
          'seaborn==0.11.2',
          'torch==1.10.0',
          'torchvision==0.11.1',
          'tensorboard==2.7.0',
          'tqdm==4.62.3',
          'wilds @ git+https://github.com/lmkoch/wilds.git'
        ],
        license='MIT')

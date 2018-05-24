from setuptools import setup, find_packages

setup(name='sac',
      version='0.0.1',
      description='sac',
      url='https://github.com/haarnoja/sac',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
